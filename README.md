# Fast-DetectGPT
**This code is for ICLR 2024 paper "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature"**, where we borrow or extend some code from [DetectGPT](https://github.com/eric-mitchell/detect-gpt).

[Paper](https://arxiv.org/abs/2310.05130) 
| [LocalDemo](#local-demo)
| [OnlineDemo](https://aidetect.lab.westlake.edu.cn/)
| [OpenReview](https://openreview.net/forum?id=Bpcgcr8E8Z)

* :fire: API support is launched. Please check the [API page](https://aidetect.lab.westlake.edu.cn/#/apidoc) in the demo.
* :fire: Fast-DetectGPT can utilize GPT-3.5 and other proprietary models as its scoring model now via [Glimpse](https://github.com/baoguangsheng/glimpse).
* :fire: So far the best sampling/scoring models we found for Fast-DetectGPT are falcon-7b/falcon-7b-instruct.

## Brief Intro
<table class="tg"  style="padding-left: 30px;">
  <tr>
    <th class="tg-0pky">Method</th>
    <th class="tg-0pky">5-Model Generations ↑</th>
    <th class="tg-0pky">ChatGPT/GPT-4 Generations ↑</th>
    <th class="tg-0pky">Speedup ↑</th>
  </tr>
  <tr>
    <td class="tg-0pky">DetectGPT</td>
    <td class="tg-0pky">0.9554</td>
    <td class="tg-0pky">0.7225</td>
    <td class="tg-0pky">1x</td>
  </tr>
  <tr>
    <td class="tg-0pky">Fast-DetectGPT</td>
    <td class="tg-0pky">0.9887 (relative↑ <b>74.7%</b>)</td>
    <td class="tg-0pky">0.9338 (relative↑ <b>76.1%</b>)</td>
    <td class="tg-0pky"><b>340x</b></td>
  </tr>
</table>
The table shows detection accuracy (measured in AUROC) and computational speedup for machine-generated text detection. The <b>white-box setting</b> (directly using the source model) is used for detecting generations produced by five source models (5-model), whereas the <b>black-box
setting</b> (utilizing surrogate models) targets ChatGPT and GPT-4 generations. AUROC results are averaged across various datasets and source models. Speedup assessments were conducted on a Tesla A100 GPU.


## Environment
* Python3.8
* PyTorch1.10.0
* Setup the environment:
  ```bash setup.sh```
  
(Notes: our experiments are run on 1 GPU of Tesla A100 with 80G memory.)

## Local Demo
Please run following command locally for an interactive demo:
```
python scripts/local_infer.py
```
where the default sampling and scoring models are both gpt-neo-2.7B.

Per impostazione predefinita, il programma utilizza `gpt-neo-2.7B` come modelli di campionamento e scoring. Per una maggiore accuratezza, è possibile utilizzare un modello di campionamento più grande (`gpt-j-6B`) con il seguente comando:
```
python scripts/local_infer.py  --sampling_model_name gpt-j-6B
```


An example (using gpt-j-6B as the sampling model) looks like
```
Please enter your text: (Press Enter twice to start processing)
Disguised as police, they broke through a fence on Monday evening and broke into the cargo of a Swiss-bound plane to take the valuable items. The audacious heist occurred at an airport in a small European country, leaving authorities baffled and airline officials in shock.

Fast-DetectGPT criterion is 1.9299, suggesting that the text has a probability of 82% to be machine-generated.
```

## Workspace
Following folders are created for our experiments:
* ./exp_main -> experiments for 5-model generations (main.sh).
* ./exp_gpt3to4 -> experiments for GPT-3, ChatGPT, and GPT-4 generations (gpt3to4.sh).

(Notes: we share <b>generations from GPT-3, ChatGPT, and GPT-4</b> in exp_gpt3to4/data for convenient reproduction.)

### Citation
If you find this work useful, you can cite it with the following BibTex entry:

    @inproceedings{bao2023fast,
      title={Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature},
      author={Bao, Guangsheng and Zhao, Yanbin and Teng, Zhiyang and Yang, Linyi and Zhang, Yue},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2023}
    }


# Documentazione in Italiano

## Fast-DetectGPT: Rilevamento Zero-Shot Efficiente di Testo Generato da Macchine

Questo repository contiene il codice per il paper ICLR 2024 "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature".

Questa versione è ottimizzata per funzionare in ambiente windows (gli script .sh sono stati convertiti in script .py).

## Introduzione

Fast-DetectGPT è uno strumento per rilevare se un testo è stato generato da un modello di linguaggio (come ChatGPT) o scritto da un essere umano. Si basa sull'analisi della curvatura della probabilità condizionale del testo.

## Requisiti di Sistema

*   Python 3.8
*   PyTorch 1.10.0
*   Una GPU NVIDIA (almeno 8GB di VRAM)

## Installazione

Per installare le dipendenze necessarie, inclusa la configurazione per la GPU, segui questi passaggi:

0. **Consiglio installare il virtual enviroment ed attivarlo**
    ```bash
    python -m venv env
    env\Scripts\activate
    ```

1.  **Disinstalla eventuali versioni precedenti di PyTorch (opzionale ma consigliato per evitare conflitti se non avete un venv):**
    ```bash
    pip uninstall torch torchvision torchaudio -y
    ```

2.  **Installa PyTorch con supporto CUDA 12.8:**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```

3.  **Installa le altre dipendenze del progetto:**
    ```bash
    pip install -r requirements.txt
    ```

## Utilizzo (Demo Locale)

È possibile avviare una demo interattiva per testare il funzionamento del programma direttamente dal tuo computer. Esegui questo comando:

```bash
python scripts/local_infer.py
```

Il programma ti mostrerà messaggi di stato come "Caricamento dei modelli..." e "Modelli caricati con successo.". Successivamente, ti chiederà di inserire un testo. Dopo averlo incollato, premi Invio due volte per avviare l'analisi. Il sistema ti restituirà un punteggio che indica la probabilità che il testo sia stato generato da una macchina.

Per una maggiore accuratezza (se avete tanta VRAM), è possibile utilizzare un modello di campionamento più grande (`gpt-j-6B`) con il seguente comando:

```bash
python scripts/local_infer.py --sampling_model_name gpt-j-6B
```

### Esempio di output:

```
Please enter your text: (Press Enter twice to start processing)
Disguised as police, they broke through a fence on Monday evening and broke into the cargo of a Swiss-bound plane to take the valuable items.

Fast-DetectGPT criterion is 1.9299, suggesting that the text has a probability of 82% to be machine-generated.
```
### Requisiti di VRAM per i Modelli (Approssimativi)

Di seguito una stima approssimativa della VRAM richiesta per i modelli più comuni. Questi valori possono variare in base alla lunghezza del testo, alla dimensione del batch e ad altri fattori.

| Modello (Parametri) | VRAM Approssimativa (GB) |
|---------------------|--------------------------|
| `gpt-neo-2.7B`      | ~5-6                     |
| `gpt-j-6B`          | ~12-14                   |
| `falcon-7b`         | ~15-18                   |
| `falcon-7b-instruct`| ~15-18                   |


### Gestione della GPU e della VRAM

Il programma è configurato per utilizzare la GPU (CUDA) per impostazione predefinita, se disponibile.

*   **Verifica CUDA:** All'avvio, il programma verifica automaticamente la disponibilità di CUDA. Se CUDA non è disponibile e hai specificato `--device cuda`, il programma ti avviserà e passerà all'elaborazione su CPU.

*   **Gestione degli errori di memoria (OOM):** Durante il caricamento dei modelli, se la VRAM disponibile non è sufficiente, il programma catturerà l'errore di "out of memory" (OOM). In questo caso, stamperà un messaggio di errore chiaro e ti suggerirà di provare a utilizzare modelli più piccoli (ad esempio, `gpt-neo-2.7B` per entrambi gli argomenti `--sampling_model_name` e `--scoring_model_name`).

*   **Selezione e Download dei Modelli:** Il programma scarica automaticamente i modelli specificati (es. `gpt-j-6B`) da Hugging Face. Se un modello è già stato scaricato in precedenza, verrà caricato dalla cache locale (`../cache` per impostazione predefinita), evitando download ripetuti. Non è necessario scaricare o posizionare manualmente i modelli. La scelta del modello deve essere fatta manualmente tramite gli argomenti `--sampling_model_name` e `--scoring_model_name`, tenendo conto della VRAM della tua GPU. Il programma non supporta direttamente il caricamento di modelli da percorsi locali specifici al di fuori della cache di Hugging Face o formati di quantizzazione come `GGUF`.

*   **Compatibilità dei Modelli per il Rilevamento:** È fondamentale notare che il rilevamento dell'AI funziona correttamente solo con le combinazioni di modelli per cui sono stati pre-calibrati i parametri di rilevamento. Queste combinazioni sono:
    *   `gpt-j-6B` (sampling) e `gpt-neo-2.7B` (scoring)
    *   `gpt-neo-2.7B` (sampling) e `gpt-neo-2.7B` (scoring)
    *   `falcon-7b` (sampling) e `falcon-7b-instruct` (scoring)
    L'utilizzo di altri modelli, anche se caricabili, non permetterà al programma di calcolare la probabilità di testo generato da AI e risulterà in un errore.

*   **Monitoraggio della VRAM:** Durante l'esecuzione, puoi monitorare l'utilizzo della VRAM con strumenti come `nvidia-smi` (su Linux/WSL) o il Task Manager di Windows (scheda "Prestazioni" -> "GPU").



### Analisi di File di Testo

È anche possibile analizzare direttamente un file di testo (`.txt`) per determinare la probabilità che il suo contenuto sia stato generato da un'intelligenza artificiale. Utilizza l'argomento `--file` seguito dal percorso del file.

Il programma analizzerà prima il testo completo, poi lo suddividerà in paragrafi (separati da doppi a capo) e analizzerà ogni porzione individualmente.

**Soglia di Rilevamento (`--threshold`):**
Puoi specificare una soglia di probabilità AI (da 0.0 a 1.0) per segnalare le porzioni di testo. Le porzioni che superano questa soglia verranno evidenziate nel report. Il valore predefinito è `0.5` (50%).

Esempio di utilizzo con soglia personalizzata:
```bash
python scripts/local_infer.py --file /percorso/del/tuo/file.txt --threshold 0.7
```

**Nota sulla codifica dei file:** Il programma si aspetta che i file di testo siano codificati in **UTF-8**. Se incontra un file con una codifica diversa, stamperà un errore e suggerirà di convertire il file in UTF-8 o di provare altre codifiche comuni.

Il programma stamperà un riepilogo sulla console e salverà un report dettagliato in un nuovo file con lo stesso nome del file originale, ma con l'aggiunta di `.analisi.txt` prima dell'estensione. Ad esempio, se analizzi `mio_testo.txt`, il report verrà salvato in `mio_testo.analisi.txt`.

L'output sulla console e nel file `.analisi.txt` sarà simile a questo:

```
Analisi del testo da /percorso/del/tuo/file.txt...
Fast-DetectGPT criterion for the whole text is 1.9299, suggesting that the text in /percorso/del/tuo/file.txt has a probability of 82% to be machine-generated.
Report di analisi salvato in: /percorso/del/tuo/file.analisi.txt
```

Il contenuto del file `.analisi.txt` includerà:

```
Fast-DetectGPT criterion for the whole text is 1.9299, suggesting that the text in /percorso/del/tuo/file.txt has a probability of 82% to be machine-generated.

--- Analisi per Paragrafi ---
Paragrafo 1 (Probabilità AI: 15%): Questo è il primo paragrafo del testo. È stato scritto da un umano.
Paragrafo 2 (Probabilità AI: 78%): Questo è il secondo paragrafo del testo. Contiene del testo generato da AI...
Paragrafo 3 (Probabilità AI: 30%): Questo è il terzo paragrafo. Anche questo è stato scritto da un umano.

--- Porzioni con probabilità AI >= 50% ---
Paragrafo 2 (Probabilità AI: 78%):
Questo è il secondo paragrafo del testo. Contiene del testo generato da AI e supera la soglia di rilevamento.
---
```

## Struttura del Progetto

*   `exp_main/`: Contiene gli esperimenti relativi a generazioni di 5 modelli di linguaggio.
*   `exp_gpt3to4/`: Contiene gli esperimenti relativi a generazioni di GPT-3, ChatGPT e GPT-4. I dati di questi esperimenti sono disponibili nella sottocartella `data` per una facile riproduzione.
*   `scripts/`: Contiene gli script Python principali, tra cui `local_infer.py` per la demo locale.
