#!/usr/bin/env python3
"""
pipeline_full.py

End-to-end pipeline for product description generation and evaluation.

This is a copy of the script included in the README, saved as a standalone file for execution.
"""
import os
import argparse
import pandas as pd
from typing import List, Tuple
import json
import time

# The following imports are used by evaluation/embedding sections.
# They require that the packages be installed in the environment.
# For safety, the script checks and raises a clear error if missing.
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge import Rouge
except Exception as e:
    raise ImportError("Missing NLTK/ROUGE dependencies. Install with: pip install nltk rouge-score") from e

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:
    SentenceTransformer = None
    st_util = None

try:
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
except Exception:
    CLIPProcessor = None
    CLIPModel = None
    Image = None

# Optional: LangChain LLM import (user will configure provider-specific LLM)
try:
    from langchain import PromptTemplate, LLMChain
except Exception:
    PromptTemplate = None
    LLMChain = None

# -----------------------------------------------------------------------------------
# Utilities: IO, validation, and preprocessing
# -----------------------------------------------------------------------------------

def load_and_validate(input_csv: str, required_cols: List[str]) -> pd.DataFrame:
    """Load CSV and validate required columns."""
    df = pd.read_csv(input_csv)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Normalize column types to strings for text columns
    for c in required_cols:
        df[c] = df[c].fillna("").astype(str)
    return df

def split_and_checkpoint(df: pd.DataFrame, output_dir: str, frac_test: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create test/reference splits and save them.

    Returns (test_df, reference_df)
    """
    test_df = df.sample(frac=frac_test, random_state=random_state)
    ref_df = df.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    test_path = os.path.join(output_dir, "test_data.csv")
    ref_path = os.path.join(output_dir, "reference_data.csv")
    test_df.to_csv(test_path, index=False)
    ref_df.to_csv(ref_path, index=False)
    print(f"Saved test -> {test_path}, reference -> {ref_path}")
    return test_df, ref_df

def normalize_metadata(metadata: str) -> str:
    """Normalize metadata string: trim, collapse spaces, ensure semicolon separators."""
    m = metadata.strip()
    # collapse multiple whitespace
    m = " ".join(m.split())
    # optional: ensure attributes use 'Key: Value; Key: Value' pattern
    # If user already provides well-structured metadata, we keep it.
    return m

# -----------------------------------------------------------------------------------
# Prompt building and generation
# -----------------------------------------------------------------------------------

DEFAULT_PROMPT = (
    "Write a creative product description (1-2 short paragraphs) for the following product metadata:\n\n"
    "{product_metadata}\n\n"
    "Guidelines:\n"
    "- Keep it concise and customer-facing (1-2 short paragraphs).\n"
    "- Mention material, fit, major attributes if present.\n"
    "- Tone: friendly and informative.\n"
)

def build_prompts(metadata_list: List[str]) -> List[str]:
    """Turn metadata strings into final prompts (strings) using DEFAULT_PROMPT."""
    prompts = []
    for m in metadata_list:
        m_norm = normalize_metadata(m)
        prompt = DEFAULT_PROMPT.format(product_metadata=m_norm)
        prompts.append(prompt)
    return prompts

def generate_descriptions_with_chain(llm, prompts: List[str], batch_size: int = 1, verbose: bool = True) -> List[str]:
    """Generate descriptions using a LangChain LLM object and PromptTemplate.

    Note: user must instantiate an LLM object compatible with LangChain and pass it in.
    Example (OpenAI):
        from langchain.llms import OpenAI
        llm = OpenAI(openai_api_key=YOUR_KEY, temperature=0.7)
        outputs = generate_descriptions_with_chain(llm, prompts)
    """
    if PromptTemplate is None or LLMChain is None:
        raise ImportError("LangChain not installed or missing. pip install langchain")

    prompt_template = PromptTemplate.from_template(DEFAULT_PROMPT)
    chain = LLMChain(prompt=prompt_template, llm=llm)

    outputs = []
    for i, prompt in enumerate(prompts):
        if verbose:
            print(f"Generating {i+1}/{len(prompts)}...")
        # Passing the metadata through the named variable product_metadata
        # (LangChain will substitute this into the template)
        try:
            # depending on langchain version, use predict or run
            text = chain.predict(product_metadata=normalize_metadata(prompt.split('\\n\\n')[1] if '\\n\\n' in prompt else prompt))
        except Exception as e:
            # fallback: try chain.run
            try:
                text = chain.run(product_metadata=normalize_metadata(prompt.split('\\n\\n')[1] if '\\n\\n' in prompt else prompt))
            except Exception as e2:
                print("LLM chain failed for an item:", e2)
                text = ""  # placeholder on failure
        outputs.append(text)
    return outputs

# -----------------------------------------------------------------------------------
# Lexical evaluation functions: BLEU, ROUGE-L, METEOR
# -----------------------------------------------------------------------------------

def evaluate_lexical(reference_texts: List[str], generated_texts: List[str]) -> Tuple[dict, List[Tuple[float,float,float]]]:
    """Compute avg BLEU, avg ROUGE-L (F1), avg METEOR, and per-example triples."""
    import nltk
    nltk.download('punkt', quiet=True)
    smoothing = SmoothingFunction().method1
    rouge = Rouge()

    bleu_scores = []
    rouge_l_scores = []
    meteor_scores = []
    per_example = []

    for ref, gen in zip(reference_texts, generated_texts):
        ref_str = (ref or "").strip()
        gen_str = (gen or "").strip()

        # tokenize
        try:
            ref_tokens = nltk.word_tokenize(ref_str.lower())
            gen_tokens = nltk.word_tokenize(gen_str.lower())
        except Exception:
            # fallback simple split
            ref_tokens = ref_str.lower().split()
            gen_tokens = gen_str.lower().split()

        # BLEU
        try:
            b = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)
        except Exception:
            b = 0.0
        bleu_scores.append(b)

        # ROUGE-L (F1)
        try:
            rouge_res = rouge.get_scores(gen_str, ref_str)[0]
            rouge_l = rouge_res['rouge-l']['f']
        except Exception:
            rouge_l = 0.0
        rouge_l_scores.append(rouge_l)

        # METEOR
        try:
            m = meteor_score([ref_str], gen_str)
        except Exception:
            m = 0.0
        meteor_scores.append(m)

        per_example.append((b, rouge_l, m))

    avg_bleu = float(sum(bleu_scores) / max(1, len(bleu_scores)))
    avg_rouge_l = float(sum(rouge_l_scores) / max(1, len(rouge_l_scores)))
    avg_meteor = float(sum(meteor_scores) / max(1, len(meteor_scores)))

    aggregate = {
        'avg_bleu': avg_bleu,
        'avg_rouge_l': avg_rouge_l,
        'avg_meteor': avg_meteor
    }
    return aggregate, per_example

# -----------------------------------------------------------------------------------
# Semantic evaluation: SentenceTransformer cosine similarity
# -----------------------------------------------------------------------------------

def compute_semantic_similarity(reference_texts: List[str], generated_texts: List[str], model_name: str = 'paraphrase-MiniLM-L6-v2', batch_size: int = 64):
    """Return average cosine similarity and per-example cosines.

    Requires sentence-transformers package.
    """
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not installed: pip install sentence-transformers")

    model = SentenceTransformer(model_name)
    emb_ref = model.encode(reference_texts, convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size)
    emb_gen = model.encode(generated_texts, convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size)
    cosine_matrix = st_util.cos_sim(emb_ref, emb_gen)
    diag = cosine_matrix.diag()
    # convert to floats
    per_example = [float(x) for x in diag.tolist()]
    avg = float(sum(per_example) / max(1, len(per_example)))
    return avg, per_example

# -----------------------------------------------------------------------------------
# CLIP image-text alignment
# -----------------------------------------------------------------------------------

def load_pil_images(image_paths: List[str], target_size=(224,224)):
    """Load images to PIL.Image with fallback placeholder."""
    imgs = []
    for p in image_paths:
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            # fallback: blank image
            img = Image.new('RGB', target_size, (255,255,255))
        imgs.append(img)
    return imgs

def compute_clip_scores(image_paths: List[str], texts: List[str], clip_model_name: str = 'openai/clip-vit-base-patch32', batch_size: int = 8):
    """Compute diagonal CLIP logits (image i vs text i). Returns list of floats."""
    if CLIPProcessor is None or CLIPModel is None:
        raise ImportError("transformers or pillow missing. Install with: pip install transformers pillow")

    import torch
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = CLIPModel.from_pretrained(clip_model_name)

    pil_images = load_pil_images(image_paths)
    # processor can batch internally; create inputs
    inputs = processor(text=texts, images=pil_images, return_tensors='pt', padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits_per_image  # (N, N)
    diag = logits.diag().tolist()
    return diag

# -----------------------------------------------------------------------------------
# Aggregation & saving
# -----------------------------------------------------------------------------------

def save_results(output_dir: str, df_input: pd.DataFrame, generated_texts: List[str],
                 lexical_per_example: List[Tuple[float,float,float]] = None,
                 semantic_per_example: List[float] = None,
                 clip_generated: List[float] = None,
                 clip_reference: List[float] = None):
    os.makedirs(output_dir, exist_ok=True)
    results = df_input.copy().reset_index(drop=True)
    results['generated_description'] = generated_texts
    if lexical_per_example is not None:
        results['bleu'] = [x[0] for x in lexical_per_example]
        results['rouge_l'] = [x[1] for x in lexical_per_example]
        results['meteor'] = [x[2] for x in lexical_per_example]
    if semantic_per_example is not None:
        results['semantic_cosine'] = semantic_per_example
    if clip_generated is not None:
        results['clip_score_generated'] = clip_generated
    if clip_reference is not None:
        results['clip_score_reference'] = clip_reference

    out_path = os.path.join(output_dir, 'generation_and_evaluation_results.csv')
    results.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")
    return out_path

# -----------------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------------

def main(args):
    required_cols = ['product_image', 'product_description', 'product_metadata']
    df = load_and_validate(args.input, required_cols)

    # 1) Split and checkpoint
    test_df, ref_df = split_and_checkpoint(df, args.output_dir, frac_test=args.frac_test, random_state=args.random_state)

    # We'll generate descriptions for the reference set (the gold set) for evaluation
    reference_texts = ref_df['product_description'].astype(str).tolist()
    metadata_list = ref_df['product_metadata'].astype(str).tolist()
    image_paths = ref_df['product_image'].astype(str).tolist()

    # 2) Build prompts
    prompts = build_prompts(metadata_list)

    # 3) GENERATION - IMPORTANT:
    if args.use_dummy:
        generated_texts = ["<DUMMY GENERATED DESCRIPTION>" for _ in prompts]
    else:
        raise ValueError("For the CLI invocation, instantiate an LLM in a Python session and call generate_descriptions_with_chain(llm, prompts). Or re-run with --use_dummy.")

    # Sanity check: lengths
    assert len(generated_texts) == len(reference_texts)

    # 4) Lexical evaluation
    agg_lex, per_example_lex = evaluate_lexical(reference_texts, generated_texts)
    print("Lexical aggregates:", agg_lex)

    # 5) Semantic evaluation (sentence-transformers)
    semantic_avg = None
    semantic_per_example = None
    if SentenceTransformer is not None:
        semantic_avg, semantic_per_example = compute_semantic_similarity(reference_texts, generated_texts)
        print("Semantic cosine avg:", semantic_avg)
    else:
        print("sentence-transformers missing; skipping semantic evaluation.")

    # 6) CLIP alignment
    clip_gen = None
    clip_ref = None
    if CLIPModel is not None and Image is not None:
        try:
            clip_gen = compute_clip_scores(image_paths, generated_texts)
            clip_ref = compute_clip_scores(image_paths, reference_texts)
            print("Computed CLIP scores (generated vs reference) for first 5:", list(zip(clip_gen[:5], clip_ref[:5])))
        except Exception as e:
            print("CLIP scoring failed:", e)
    else:
        print("transformers/CLIP or pillow not installed; skipping CLIP evaluation.")

    # 7) Save results
    out_path = save_results(args.output_dir, ref_df, generated_texts,
                            lexical_per_example=per_example_lex,
                            semantic_per_example=semantic_per_example,
                            clip_generated=clip_gen,
                            clip_reference=clip_ref)

    # 8) Print final summary
    print("=== SUMMARY ===")
    print("Avg BLEU:", agg_lex['avg_bleu'])
    print("Avg ROUGE-L:", agg_lex['avg_rouge_l'])
    print("Avg METEOR:", agg_lex['avg_meteor'])
    if semantic_avg is not None:
        print("Avg Semantic Cosine:", semantic_avg)
    if clip_gen is not None and clip_ref is not None:
        diffs = [g - r for g, r in zip(clip_gen, clip_ref)]
        print("Mean CLIP (generated - reference):", float(sum(diffs) / len(diffs)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Product description generation & evaluation pipeline')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV with required columns')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save outputs')
    parser.add_argument('--frac_test', type=float, default=0.2, help='Fraction of data to reserve as test')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--use_dummy', action='store_true', help='If set, generate dummy placeholder texts instead of calling an LLM')
    args = parser.parse_args()
    main(args)
