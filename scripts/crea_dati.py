
import json
import os

def prepara_dati_per_analisi(cartella_input, file_output):
    """
    Legge i file di testo da una cartella, li formatta in JSON e li salva.

    Args:
        cartella_input (str): Il percorso della cartella contenente i file .txt.
        file_output (str): Il percorso del file JSON di output.
    """
    dati = {"original": [], "sampled": []}
    
    # Assumiamo che i file .txt siano il testo "campionato" (sampled)
    # e usiamo un testo fittizio per l'"originale" come richiesto dallo script.
    testo_originale_fittizio = "Questo è un testo segnaposto scritto da un umano."

    for nome_file in os.listdir(cartella_input):
        if nome_file.endswith(".txt"):
            percorso_file = os.path.join(cartella_input, nome_file)
            with open(percorso_file, 'r', encoding='latin-1') as f:
                contenuto = f.read()
                dati["sampled"].append(contenuto)
                # Aggiungiamo un testo originale fittizio per ogni file campionato
                dati["original"].append(testo_originale_fittizio)

    with open(file_output, 'w', encoding='utf-8') as f:
        json.dump(dati, f, ensure_ascii=False, indent=4)

    print(f"Dati salvati in {file_output}")

if __name__ == '__main__':
    cartella_input = './file'
    file_output = './dati_analisi.json'
    prepara_dati_per_analisi(cartella_input, file_output)
