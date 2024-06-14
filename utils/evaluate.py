from nltk.translate import bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import normalizeString
# from utils.preprocess import normalizeString


def bleu_evaluate(candidates: str, references: str):
    candidates = normalizeString(candidates)
    references = normalizeString(references)
    candidates_list = candidates.split()
    references_list = references.split()

    return bleu([candidates_list], references_list)


def rouge_evaluate(candidates: str, references: str):
    candidates = normalizeString(candidates)
    references = normalizeString(references)

    rouge_ = Rouge()
    return rouge_.get_scores(hyps=candidates, refs=references, avg=True)


def bert_evaluate(candidates: str, references: str):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    sentence = [candidates, references]

    embedding = model.encode(sentence)
    cos_sim = cosine_similarity([embedding[0]],[embedding[1]])

    return cos_sim.item()


if __name__ == '__main__':
    candidates = "i am a student from xx school. happy new year"
    references = "i am a student from school on china. happy birthday"
    rouge_score = bert_evaluate(references, references)
