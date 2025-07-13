# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import os
import glob
import argparse
import json
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
from scipy.stats import norm


# Considering balanced classification that p(D0) equals to p(D1), we have
#   p(D1|x) = p(x|D1) / (p(x|D1) + p(x|D0))
def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob

class FastDetectGPT:
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic

        try:
            self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
            self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
            self.scoring_model.eval()

            if args.sampling_model_name != args.scoring_model_name:
                self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
                self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
                self.sampling_model.eval()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"\nErrore durante il caricamento del modello: {e}")
            print("Potrebbe essere un problema di memoria GPU insufficiente.")
            print("Suggerimento: Prova a usare modelli più piccoli, ad esempio:")
            print("  --sampling_model_name gpt-neo-2.7B --scoring_model_name gpt-neo-2.7B")
            exit(1) # Termina il programma se non riesce a caricare i modelli
        # To obtain probability values that are easy for users to understand, we assume normal distributions
        # of the criteria and statistic the parameters on a group of dev samples. The normal distributions are defined
        # by mu0 and sigma0 for human texts and by mu1 and sigma1 for AI texts. We set sigma1 = 2 * sigma0 to
        # make sure of a wider coverage of potential AI texts.
        # Note: the probability could be high on both left side and right side of Normal(mu0, sigma0).
        #   gpt-j-6B_gpt-neo-2.7B: mu0: 0.2713, sigma0: 0.9366, mu1: 2.2334, sigma1: 1.8731, acc:0.8122
        #   gpt-neo-2.7B_gpt-neo-2.7B: mu0: -0.2489, sigma0: 0.9968, mu1: 1.8983, sigma1: 1.9935, acc:0.8222
        #   falcon-7b_falcon-7b-instruct: mu0: -0.0707, sigma0: 0.9520, mu1: 2.9306, sigma1: 1.9039, acc:0.8938
        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'gpt-neo-2.7B_gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
        }
        key = f'{args.sampling_model_name}_{args.scoring_model_name}'
        self.classifier = distrib_params[key]

    # compute conditional probability curvature
    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.sampling_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    # compute probability
    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob, crit, ntoken


# run interactive local inference
def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Attenzione: CUDA non disponibile. L'elaborazione avverrà su CPU.")
        args.device = "cpu"

    print(f"Caricamento dei modelli {args.sampling_model_name} e {args.scoring_model_name}...")
    detector = FastDetectGPT(args)
    print("Modelli caricati con successo.")

    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
            if len(text) == 0:
                print(f"File {args.file} is empty.")
                return
            print(f"Analisi del testo da {args.file}...")

            # estimate the probability of machine generated text for the whole text
            prob_full, crit_full, ntokens_full = detector.compute_prob(text)
            result_string_full = f'Fast-DetectGPT criterion for the whole text is {crit_full:.4f}, suggesting that the text in {args.file} has a probability of {prob_full * 100:.0f}% to be machine-generated.'
            print(result_string_full)

            report_content = [result_string_full, ""]

            # Analyze text by paragraphs
            paragraphs = text.split('\n\n')
            flagged_paragraphs = []
            
            report_content.append("--- Analisi per Paragrafi ---")
            for i, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                prob_para, crit_para, ntokens_para = detector.compute_prob(paragraph)
                paragraph_result = f"Paragrafo {i+1} (Probabilità AI: {prob_para * 100:.0f}%): {paragraph[:100]}..." # Show first 100 chars
                report_content.append(paragraph_result)

                if prob_para * 100 >= args.threshold * 100:
                    flagged_paragraphs.append({
                        "index": i + 1,
                        "probability": prob_para * 100,
                        "text": paragraph
                    })
            report_content.append("")

            if flagged_paragraphs:
                report_content.append(f"--- Porzioni con probabilità AI >= {args.threshold * 100:.0f}% ---")
                for fp in flagged_paragraphs:
                    report_content.append(f"Paragrafo {fp['index']} (Probabilità AI: {fp['probability']:.0f}%):")
                    report_content.append(fp['text'])
                    report_content.append("---")
            else:
                report_content.append(f"Nessuna porzione ha superato la soglia del {args.threshold * 100:.0f}% di probabilità AI.")

            # Save report to a file
            base, ext = os.path.splitext(args.file)
            output_file_path = base + ".analisi" + ext
            try:
                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    outfile.write('\n'.join(report_content) + '\n')
                print(f"Report di analisi salvato in: {output_file_path}")
            except IOError as e:
                print(f"Errore durante il salvataggio del report di analisi in {output_file_path}: {e}")

        except FileNotFoundError:
            print(f"Errore: File non trovato a {args.file}")
        except UnicodeDecodeError:
            print(f"Errore di codifica: Impossibile leggere il file {args.file} con la codifica UTF-8.")
            print("Assicurati che il file sia codificato in UTF-8. Potresti provare altre codifiche come 'latin-1' o 'cp1252' se conosci l'encoding originale del file.")
            print("Per convertire un file, puoi usare un editor di testo come Notepad++ o VS Code, oppure strumenti da riga di comando.")
        except Exception as e: # Catch any other unexpected errors during analysis
            print(f"Si è verificato un errore inatteso durante l'analisi del file: {e}")
        return

    # input text
    print('Local demo for Fast-DetectGPT, where the longer text has more reliable result.')
    print('')
    while True:
        print("Please enter your text: (Press Enter twice to start processing)")
        lines = []
        while True:
            line = input()
            if len(line) == 0:
                break
            lines.append(line)
        text = "\n".join(lines)
        if len(text) == 0:
            break
        # estimate the probability of machine generated text
        prob, crit, ntokens = detector.compute_prob(text)
        print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be machine-generated.')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None, help="Path to a .txt file to analyze.")
    parser.add_argument('--sampling_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for flagging AI-generated portions (0.0 to 1.0).")
    args = parser.parse_args()

    run(args)




