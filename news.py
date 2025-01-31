from pathlib import Path
import datasets
import torch

FAISS_INDEX_PATH = Path('local/news-please/faiss_index')

class DPR():
    def __init__(self):
        self.ds = datasets.load_dataset('sanxing/advfake_news_please')['train']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.index_dpr()

    @torch.no_grad()
    def index_dpr(self):
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

        faiss_path = FAISS_INDEX_PATH / 'my_index.faiss'
        if faiss_path.exists():
            print('loading faiss index')
            self.ds.load_faiss_index('embeddings', str(faiss_path))
            return

        ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(self.device)
        ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        ds_with_embeddings = self.ds.map(lambda example: {
            'embeddings': ctx_encoder(**ctx_tokenizer(example["title"], return_tensors="pt", padding=True).to(self.device))[0].cpu().numpy()
        }, batched=True, batch_size=64)
        ds_with_embeddings.add_faiss_index(column='embeddings')


        print('saving faiss index')
        ds_with_embeddings.save_faiss_index('embeddings', str(faiss_path))

    def interactive(self):
        while True:
            question = input('Enter a question: ')
            if question == 'exit':
                break
            question = question.strip()
            if not question:
                continue
            scores, retrieved_examples = self.search(question)
            for idx, (score, example) in enumerate(zip(scores, retrieved_examples)):
                print(f'{idx} - {score:.2f} - {example["title"]}')
                print(f'{example["date_publish"]} - {example["url"]}')
                print(f'{example["description"]}\n\n')
            print('---')

    @torch.no_grad()
    def search(self, question):
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer


        q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

        question_embedding = q_encoder(**q_tokenizer(question, return_tensors="pt"))[0][0].numpy()
        scores, retrieved_examples = self.ds.get_nearest_examples('embeddings', question_embedding, k=10)
        retrieved_examples = [dict(zip(retrieved_examples,t)) for t in zip(*retrieved_examples.values())]

        return scores, retrieved_examples


if __name__ == '__main__':
    dpr = DPR()
    dpr.interactive()