from pathlib import Path
import argparse

import datasets
from thefuzz import fuzz

from news import DPR
from transformers import logging
from utils import get_gpt_response, get_gemini_response
from search import get_google_ctx

logging.set_verbosity_error() 


DATASET_PATH = "local/hf_datasets/"
GPT4 = "gpt-4o-2024-05-13"
REWRITE_THRESHOLD = 60

def get_chat_preds(messages, model, cot, stop=None):
    try:
        if model.startswith("gpt"):
            res = get_gpt_response(messages, model, 1.0, 3 if cot else 100, 1000 if cot else 10)
            preds = [choice.message.content.splitlines()[-1] for choice in res.choices]
        elif model.startswith("gemini"):
            res = get_gemini_response(messages, model, 1.0, 1, 1000 if cot else 10, stop)
            preds = [res.splitlines()[-1]]
        elif model.startswith("meta"):
            res = get_gpt_response(messages, model, 1.0, 3 if cot else 6, 1000 if cot else 10, stop=["<|eot_id|>","<|eom_id|>"])
            preds = [choice.message.content.splitlines()[-1] for choice in res.choices]
    except Exception as e:
        print(e)
        preds = []
    return preds


def process_numerical_preds(predictions):
    # filter out non-numeric predictions
    predictions = [
        p.replace(".", "") for p in predictions if p.replace(".", "").isdigit()
    ]
    # keep only 1-10
    predictions = [int(p) for p in predictions if 1 <= int(p) <= 10]

    if len(predictions) == 0:
        print("No valid predictions")
        return 0, [], 0, 0
    
    majority = max(set(predictions), key=predictions.count)
    
    score = sum(predictions) / len(predictions)
    variance = sum([(p - score) ** 2 for p in predictions]) / len(predictions)
    return score, predictions, variance, majority


def get_retrieval_ctx(example, prefix, source="dpr"):
    cnt = 0
    text = "Related news stories from search results:\n\n"
    if source == 'google':
        text += get_google_ctx(example[prefix + "title"]) + "\n\n"
    elif source == 'dpr':
        for rex in example[prefix + "dpr_retrieved_examples"]:
            if rex["url"] == example["url"]:
                # skip the example itself
                continue
            text += f'{rex["date_publish"].date()} - {rex["title"]}\n{rex["url"]}\n{rex["description"]}\n\n'
            cnt += 1
            if cnt == 5:
                break
    else:
        raise ValueError("Invalid source")
    return text


def get_score(example, rag, prefix="", rationale=False, model=GPT4):
    """
    Score the plausicility of a news story.

    Args:
        example: The example to get the score for.
        rag: Whether to use RAG context.
        prefix: Empty string or "f_" for fake news.
        rationale: Whether to ask for rationale (Chain of Thought).
        model: The model to use.
    Returns:
        The mean, the variance, and the raw predictions of all candidate scores.
    """

    text = get_retrieval_ctx(example, prefix) if rag else ""
    text += "Please predict the plausibility of the following news story:\n\n"
    text += f'{example["date_publish"].date()} - {example[prefix + "title"]}\n{example[prefix + "description"]}\n\n'

    predictions = get_chat_preds(
        messages=[
            {
                "role": "system",
                "content": "Today is March 26, 2024. You predict the plausibility of a news you haven't seen" + (", given a list of related news stories from search results." if rag else "."),
            },
            (
                {
                    "role": "user",
                    "content": text
                    + "\n\nPlease give a number representing the plausibility (1-10) and nothing else.",
                }
                if not rationale
                else {
                    "role": "user",
                    "content": text
                    + "\n\nPlease first give me your reasoning, and start a new line to give a number representing the plausibility (1-10) and nothing else.",
                }
            ),
        ],
        model=model,
        cot=rationale,
    )

    score, predictions, variance, majority = process_numerical_preds(predictions)

    suffix = ""

    if rag:
        prefix += "rag_"

    return {
        prefix + "score" + suffix: score,
        prefix + "preds" + suffix: predictions,
        prefix + "var" + suffix: variance,
        prefix + "majority" + suffix: majority,
    }


def get_rationale(example, rag, prefix=""):

    text = get_retrieval_ctx(example, prefix) if rag else ""
    text += "Please predict the plausibility of the following news story:\n\n"
    text += f'{example["date_publish"].date()} - {example[prefix + "title"]}\n{example[prefix + "description"]}\n\n'

    res = get_gpt_response(
        messages=[
            {
                "role": "system",
                "content": "Today is March 26, 2024. You fact-check a news you haven't seen, given a list of related news stories from search results.",
            },
            {
                "role": "user",
                "content": text
                + "\n\nPlease explain what you find suspicious about the news story. Give the top 3 points in a list format and nothing else.",
            },
        ],
        model=GPT4,
        temperature=0,
    )

    rationale = res.choices[0].message.content.strip()

    return {
        prefix + ("rag_" if rag else "") + "rationale": rationale,
    }


def confirm_fake(title, description, f_title, f_description):

    title_similarity = fuzz.ratio(f_title, title)
    description_similarity = fuzz.ratio(f_description, description)
    if (
        title_similarity <= REWRITE_THRESHOLD
        or description_similarity <= REWRITE_THRESHOLD
    ):
        return False, [
            "Fail to satisfy rewrite similarity threshold",
            title_similarity,
            description_similarity,
        ]

    text = f"News 1\nTitle: {title}\nDescription: {description}\n\nNews 2\nTitle: {f_title}\nDescription: {f_description}"

    res = get_gpt_response(
        messages=[
            {
                "role": "user",
                "content": text
                + "\n\nDoes News 2 conflict with News 1? Please type yes or no and nothing else.",
            },
        ],
        model=GPT4,
        temperature=0.0,
        n=10,
    )

    raw_output = [c.message.content.lower() for c in res.choices]
    ratio = sum(["yes" in c.message.content.lower() for c in res.choices]) / len(
        res.choices
    )

    return ratio >= 0.8, raw_output


def adv_rewrite(example, rag=False):

    text = "Please rewrite the following news story.\n\n"
    # date - title - description
    text += f'News Date: {example["date_publish"].date()}\nTitle: {example["title"]}\nDescription: {example["description"]}\n\n'
    text += f'It was previously rewrote as:\nTitle: {example["f_title"]}\nDescription: {example["f_description"]}\n\n'
    rationale = example["f_rag_rationale"] if rag else example["f_rationale"]
    text += f'Fact-checkers found the following suspicious:\n{rationale}\n\n'
    text += f"Please rewrite the news story to make it more believable and fool the fact-checkers."

    date_instructions = "Today is March 26, 2024."
    open_instructions = " Please rewrite a real recent news story to introduce some believable factual errors. The modification should be able to fool professional fact-checkers."
    format_instructions = "\n\nFormat\nError introduced:\nTitle:\nDescription:"

    instructions = date_instructions + open_instructions + format_instructions

    samples = []
    error_message = ["Failed to rewrite:", text]

    for _ in range(5):
        res = get_gpt_response(
            messages=[
                {
                    "role": "system",
                    "content": instructions,
                },
                {"role": "user", "content": text},
            ],
            model=GPT4,
            temperature=1.0,
            n=8,
            stop=["Fact-checkers"],
            max_tokens=300,
        )

        for i in range(len(res.choices)):
            content = res.choices[i].message.content
            error_message.append(content)
            # try parse the title and description, otherwise return empty strings
            try:
                content = content.replace(":\n", ": ")
                items = content.strip().split("\n")
                items = [
                    item for item in items if item and not item.startswith("News Date:")
                ]
                error, title, description = items[:3]
                title = title[7:] if title.startswith("Title: ") else title
                description = (
                    description[13:]
                    if description.startswith("Description: ")
                    else description
                )
                error = error[18:] if error.startswith("Error introduced: ") else error

                success, raw_output = confirm_fake(
                    example["title"], example["description"], title, description
                )
                if success:
                    samples.append(
                        (
                            get_score(example, True, "f_", rationale=False)[
                                "f_rag_score"
                            ],
                            {
                                "f_title": title,
                                "f_description": description,
                                "f_error": error,
                            },
                        )
                    )
                else:
                    error_message.append(str(raw_output))
            except Exception as e:
                error_message.append(str(e))
                continue

        if len(samples) > 0:
            break

    if len(samples) == 0:
        return {
            "f_title": "",
            "f_description": "",
            "f_error": "",
        }
    else:
        # return the one with the highest score
        return max(samples, key=lambda x: x[0])[1]


def rewrite(example, rag=False, type="entity"):

    text = (
        (
            get_retrieval_ctx(example, prefix="f_")
            + f"Please rewrite the following news story.\n\n"
        )
        if rag
        else ""
    )
    # date - title - description
    text += f'News Date: {example["date_publish"].date()}\nTitle: {example["title"]}\nDescription: {example["description"]}'

    date_instructions = "Today is March 26, 2024."
    subs_instructions = " You will be given a recent news story, please rewrite it by substituting one or two entities (names or locations) to their equivalence, but the news should still looks true."
    open_instructions = " Please rewrite a real recent news story to introduce some believable factual errors. The modification should be able to fool professional fact-checkers."
    rag_instructions = (
        " You should also consider a list of related news stories from search results that people might use to fact-check the news story you write."
        if rag
        else ""
    )
    format_instructions = "\n\nFormat\nError introduced:\nTitle:\nDescription:"

    instructions = (
        date_instructions
        + (subs_instructions if type == "entity" else open_instructions)
        + rag_instructions
        + format_instructions
    )

    for _ in range(5):
        res = get_gpt_response(
            messages=[
                {
                    "role": "system",
                    "content": instructions,
                },
                {"role": "user", "content": text},
            ],
            model=GPT4,
            temperature=1.0,
            n=5,
        )

        for i in range(len(res.choices)):
            content = res.choices[i].message.content
            # try parse the title and description, otherwise return empty strings
            try:
                content = content.replace(":\n", ": ")
                items = content.strip().split("\n")
                items = [item for item in items if item]
                error, title, description = items
                title = title[7:] if title.startswith("Title: ") else title
                description = (
                    description[13:]
                    if description.startswith("Description: ")
                    else description
                )
                error = error[18:] if error.startswith("Error introduced: ") else error
                if confirm_fake(
                    example["title"], example["description"], title, description
                ):
                    return {
                        "f_title": title,
                        "f_description": description,
                        "f_error": error,
                    }
            except:
                continue

    print("Failed to rewrite")
    return {
        "f_title": "",
        "f_description": "",
        "f_error": "",
    }


def get_dpr_results(example, dpr, search_key="title", prefix=""):
    scores, retrieved_examples = dpr.search(example[prefix + search_key])

    # store score to each example
    for idx, (score, rex) in enumerate(zip(scores, retrieved_examples)):
        rex["dpr_score"] = score

    # store the index of the example in the retrieved examples
    recall_idx = -1
    for idx, (score, rex) in enumerate(zip(scores, retrieved_examples)):
        if rex["url"] == example["url"]:
            recall_idx = idx
            break

    return {
        prefix + "dpr_retrieved_examples": retrieved_examples,
        prefix + "dpr_recall_idx": recall_idx,
    }


def get_new_dataset(ds, args):
    if args.preflight:
        # get the first n examples
        n = 10
        ds = ds.select(range(n))
        print("Preflight check with {n} examples".format(n=n))

    print("=" * 80)
    print("Initiating RAG")
    dpr = DPR()
    print("=" * 80)

    if args.first_round:
        # get score for positive examples
        ds = ds.map(lambda example: get_score(example, rag=False), num_proc=args.num_proc)
        print("Scored positive examples")

        # get DPR results for positive examples, num_proc=1 is important
        ds = ds.map(lambda example: get_dpr_results(example, dpr), num_proc=1)

        # get rag score for positive examples
        ds = ds.map(
            lambda example: get_score(example, rag=True, rationale=False),
            num_proc=args.num_proc,
        )
        print("Scored positive examples w/ RAG")

    shift = 1
    for round_num in range(shift, args.num_rounds + shift):

        print(f"Round {round_num}")

        if args.generation_context_type == "none":
            # generate negative examples (fake news)
            ds = ds.map(
                lambda example: rewrite(
                    example, rag=False, type=args.substitution_type
                ),
                num_proc=args.num_proc,
            )
            print("Generated negative examples")

        elif args.generation_context_type == "rag_raw":
            # generate negative examples (fake news) w/ DPR
            ds = ds.map(
                lambda example: rewrite(example, rag=True, type=args.substitution_type),
                num_proc=args.num_proc,
            )
            print("Generated negative examples w/ retrieval context")

        elif args.generation_context_type == "rag_rationale":
            ds = ds.map(
                lambda example: adv_rewrite(
                    example, rag=True
                ),
                num_proc=args.num_proc,
            )
            print("Generated negative examples w/ detector rationale (RAG)")

        elif args.generation_context_type == "rationale":
            ds = ds.map(
                lambda example: adv_rewrite(
                    example, rag=False
                ),
                num_proc=args.num_proc,
            )
            print("Generated negative examples w/ detector rationale")

        # filter out examples that are not rewritten, i.e., empty f_title or f_description
        size_before_filter = ds.num_rows
        ds = ds.filter(
            lambda example: len(example["f_title"]) > 0
            and len(example["f_description"]) > 0
        )
        print(
            f"Filtered out {size_before_filter - ds.num_rows} examples that could not be rewritten."
        )

        # get score for negative examples
        ds = ds.map(
            lambda example: get_score(example, rag=False, prefix="f_"), num_proc=args.num_proc
        )
        print("Scored negative examples")

        # get DPR results for negative examples
        ds = ds.map(
            lambda example: get_dpr_results(example, dpr, prefix="f_"), num_proc=1
        )

        # get rag score for negative examples
        ds = ds.map(
            lambda example: get_score(example, rag=True, prefix="f_", rationale=False),
            num_proc=args.num_proc,
        )
        print("Scored negative examples w/ RAG")

        # get post-hoc rationale (w/o RAG) for negative examples
        ds = ds.map(
            lambda example: get_rationale(example, False, "f_"), num_proc=args.num_proc
        )
        # get post-hoc rationale (w/ RAG) for negative examples
        ds = ds.map(
            lambda example: get_rationale(example, True, "f_"), num_proc=args.num_proc
        )

        print(f"Round {round_num} completed")
        print(f'ROC AUC: {get_roc_auc(ds["score"], ds["f_score"])}')
        print(f'ROC AUC (RAG): {get_roc_auc(ds["rag_score"], ds["f_rag_score"])}')

        # save the dataset to disk
        ds.save_to_disk(str(args.path) + f"_round{round_num}")
    return ds


def get_roc_auc(positives, negatives):
    from sklearn import metrics

    probs = list(positives) + list(negatives)
    preds = [1] * len(positives) + [0] * len(negatives)
    fpr, tpr, thresholds = metrics.roc_curve(preds, probs)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


if __name__ == "__main__":
    """
    DISCLAIMER: This code is for research purposes only, specifically for studying misinformation detection
    and improving fact-checking systems. Any use of this code for generating and spreading actual 
    misinformation is strictly prohibited and unethical.
    
    RESPONSIBLE USAGE:
    - Use only for academic research and improving detection systems
    - Do not deploy for generating actual fake news
    - Follow ethical guidelines for AI research
    """
    # create a parser
    parser = argparse.ArgumentParser(
        description="Research tool for studying misinformation detection through adversarial examples."
    )
    # add arguments to the parser
    parser.add_argument(
        "--source",
        type=str,
        help="The source dataset to be used.",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="The target dataset to be created.",
        required=True,
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Whether to run the preflight check on about 10 examples.",
    )
    parser.add_argument(
        "--first-round",
        action="store_true",
        help="Whether to run the first round of the game. Only in the first round, we score the positives.",
    )
    parser.add_argument(
        "--substitution-type",
        type=str,
        default="open",
        help="The type of generation: entity, open",
    )
    parser.add_argument(
        "--generation-context-type",
        type=str,
        default="none",
        help="The type of context: none, rag_raw, rag_rationale",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="The number of rounds to run the game.",
    )
    parser.add_argument(
        "--num-proc", type=int, default=10, help="The number of processors to use."
    )

    # parse the arguments
    args = parser.parse_args()


    if not args.source:
        ds = datasets.load_dataset('sanxing/advfake')
        # convert the date_publish to timestamp
        import datetime
        ds = ds['train'].map(lambda x: {'date_publish_timestamp': datetime.datetime.strptime(x['date_publish'], '%Y-%m-%d %H:%M:%S')})
        # drop the date_publish column
        ds = ds.remove_columns('date_publish')
        # change the date_publish_timestamp to date_publish
        ds = ds.rename_column('date_publish_timestamp', 'date_publish')

    else:
        source_path = DATASET_PATH + args.source
        # verify path
        if not Path(source_path).exists():
            raise ValueError(f"{DATASET_PATH + args.source} does not exist.")
        ds = datasets.load_from_disk(source_path)

    # if directory exists, give error
    args.path = Path(f"{DATASET_PATH}{args.target}")
    if args.path.exists():
        raise ValueError(f"{args.path} already exists.")

    ds = get_new_dataset(ds, args)
