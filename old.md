# Comandi utilizzati per l'installazione dei pacchetti e delle dipendenze

```sh
pip3 install thop
pip3 install libtiff
pip3 install timm
pip3 install opencv-python
pip3 install scipy
pip3 install numpy
pip3 install nibabel
pip3 install nnunet
pip3 install matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
<br/><br/>

# Modifiche preliminari per far funzionare il codice

* Modificato il file `pvt.py`.
Nella classe `class HSNet(nn.Module):` è stato modificato il costruttore `def __init__(self, channel=32)`in questo modo:

    ```python
        from
            path = './pretrained_pth/pvt_v2_b2.pth'
        to
            path = './pre-trained/pvt_v2_b2.pth'
    ```
<br/>

* Sono stati cambiati i percorsi (path) nel file ``train.py`` per il training ed il test in base a dove sono posizionate le cartelle nel mio computer.
<br/>

* Ridotto il batch size da 8 a 2 nel file ``train.py`` per evitare il seguente errore:

      torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 4.00 GiB total capacity; 3.36 GiB already allocated; 0 bytes free; 3.48 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
      See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
<br/>

* Modificato il file `train.py`.
Nella funzione `def train(train_loader, model, optimizer, epoch, test_path)` è stato modificato il path in questo modo:

    ```python
        from
            test1path = './dataset/TestDataset/'
        to
            test1path = '../dataset/TestDataset'
    ```
<br/>

* Modificato il file `train.py`.
Aggiunto un print alla fine della fase di train per indicare quando ha concluso.

    ```python

        print("#" * 20, "  End Training  ", "#" * 20)

    ```
<br/>

* Ridotto il numero di epoch da 100 a 2 nel file ``train.py`` per velocizzare il tutto e vedere se conclude.

<br/><br/>

# Inizio lavorazioni

* Eliminato la funzione structure_loss dal file ``train.py`` in quanto già presente nel file ``loss.py`. 
<br/>

* Sistemato secondo le google docstrings gli import del file ``train.py``.
<br/>

* Importato tutte le loss presenti nel file ``loss.py`` nel file ``train.py``.

    ```python
        # Loss already present in the file
        from loss import bce_loss, dice_loss, IoU_loss
        from loss import dice_bce_loss, log_cosh_dice_loss
        from loss import focal_loss, tversky_loss
        from loss import focal_tversky_loss, combo_loss
        from loss import structure_loss
    ```
<br/>

* Creato il file copia di ``loss.py`` chiamato ``lossTest.py``. Questo file conterrà tutte le nuove modifiche e le nuove loss così da non toccare il file originale.
<br/>

* Come indicato ho commentato la riga:

    ```python
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    ```
    In quanto altrimenti si troverà sempre una foreground region.
    Questa modifica è stata fatta nei file:

    * ``Train.py``
    * ``Test.py``
    * ``HarDNet-MSEG.py``
    * ``HSNet.py``
<br/>

* Commentato tutto il codice del file ``lossTest.py`` in stile Google Docstrings
<br/>

* Commentato tutto il codice del file ``Train.py`` in stile Google Docstrings
<br/>

* Aggiornato le funzioni ``train`` e ``test`` nel file ``Train.py`` per interrompere l'avviso "nn.functional.upsample is deprecated
    Use nn.functional.interpolate instead." che si verifica quando usiamo la funzione upsample di PyTorch. La soluzione era sostituire F.upsample con F.interpolate.

    Esempio dell'errore:

    ```sh

        \\YOUR-PATH\HSNet\PyTorch\env\Lib\site-packages\torch\nn\functional.py:3737:
        UserWarning: nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.
        warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")

    ```

    Per ulteriori informazioni sull'errore vedere ne note sulle 2 funzioni indicate.
<br/>

* Modificato i percorsi delle cartelle nel file ``Test.py`` (Che è uguale identico al file ``HSNet.py``)

    ```python
        from
            data_path = '/dataset/TestDataset/{}'.format(_data_name)
        to
            data_path = '../dataset/TestDataset/{}'.format(_data_name)
        
    ```
<br/>

* Sistemato secondo le google docstrings gli import del file ``Test.py``.

<br/>

* Aggiornato il file ``Test.py``  per interrompere l'avviso "nn.functional.upsample is deprecated
    Use nn.functional.interpolate instead." che si verifica quando usiamo la funzione upsample di PyTorch. La soluzione era sostituire F.upsample con F.interpolate.
    Stesso procedimento fatto in precedenza per ulteriori informazioni vedi sopra.

<br/>

* Sostituito momentaneamente la seguente riga nel file ``Test.py``:

    ```python
        from
            parser.add_argument('--pth_path', type=str, default='./model_pth/HSNet.pth')
    
        to
            parser.add_argument('--pth_path', type=str, default='./model_pth/PolypPVT/PolypPVT.pth')
        
    ```

    Questo percè non ho quella ``HSNet.pth`` e non so dove trovarla.

<br/>

### NOTA:
Ho chiesto chiarimenti sul da farsi e bisogna implementare questa loss [Region-wise Loss](https://arxiv.org/abs/2108.01405) sostituendola con quella presente nel file ``Train.py``.

<br/>

* Importato il file ``rwexp_loss.py``.

<br/>

* Per comodità ho copiato la classe ``RWLoss()`` presente nel file ``rwexp_loss.py`` nel file ``lossTest.py`` con tutte le altre.

<br/>

* Commentato ogni passaggio della funzione ``forward(self, x, y_)``  della classe ``RWLoss()`` presente nel file ``lossTest.py``.

<br/>

* Aggiunto la funzione ``rw_loss.forward(P1, gts)`` al file ``Train.py`` e commentato le altre funzioni non necessarie. Ho inoltre commentato **MOMENTANEAMENTE** altre parti come si può vedere per evitare ulteriori e concentrarmi sull'errore principale sottostante che bisogna risolvere:

    ```python
        from
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            loss = loss_P1 + loss_P2 + loss_P3 + loss_P4
            
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P4.data, opt.batchsize)
    
        to
            # ---- loss function ----
            # loss_P1 = structure_loss(P1, gts)
            # loss_P2 = structure_loss(P2, gts)
            # loss_P3 = structure_loss(P3, gts)
            # loss_P4 = structure_loss(P4, gts)
            # loss = loss_P1 + loss_P2 + loss_P3 + loss_P4
            # Creazione dell'istanza della classe RWLoss
            rw_loss = RWLoss()
            loss = rw_loss.forward(gts, P1)
            
            # ---- backward ----
            # loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            # ---- recording loss ----
            if rate == 1:
                #loss_P2_record.update(loss_P4.data, opt.batchsize)
                loss_P2_record.update(loss.data, opt.batchsize)
        
    ```

<br/>

* Una volta fatto installate le dipendenze andare sul file ``./env/Lib/site-packages/nnunet/__init__.py`` e commentare le seguenti righe di codice altrimenti spaunano ogni 10 secondi dando fastidio e basta:

    ```python

    print("\n\nPlease cite the following paper when using nnUNet:\n\nIsensee, F., Jaeger, P.F., Kohl, S.A.A. et al. "
        "\"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.\" "
        "Nat Methods (2020). https://doi.org/10.1038/s41592-020-01008-z\n\n")
    print("If you have questions or suggestions, feel free to open an issue at https://github.com/MIC-DKFZ/nnUNet\n")

    ```

<br/>

* Aggiunto il file ``folder_path.py`` così da non dover cambiare ogni volta nel codice i percorsi in base a dove viene runnato il codice. Basta semplicemente aggiungere le righe nel file e commentare le altre. 

<br/>

* ...

<br/><br/><br/><br/><br/><br/>

# ERRORE DA RISOLVERE ADESSO
Questo è l'errore sul quale si sta lavorando al momento:

```sh
AdamW (
Parameter Group 0        
    amsgrad: False       
    betas: (0.9, 0.999)  
    capturable: False    
    differentiable: False
    eps: 1e-08
    foreach: None        
    fused: None
    lr: 0.0001
    maximize: False      
    weight_decay: 0.0001 
)
True
no augmentation
#################### Start Training ####################
C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\cuda\ScatterGatherKernel.cu:367: block: [55,0,0], thread: [64,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed.
...
...
...
C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\cuda\ScatterGatherKernel.cu:367: block: [43,0,0], thread: [63,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed.
Traceback (most recent call last):
  File "\\NAS_home\Develop\Coding\Research\lossFuncition-4SN\main\Train.py", line 348, in <module>

  File "\\NAS_home\Develop\Coding\Research\lossFuncition-4SN\main\Train.py", line 184, in train
    #loss = loss_P1 + loss_P2 + loss_P3 + loss_P4
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "\\NAS_home\Develop\Coding\Research\lossFuncition-4SN\main\lossTest.py", line 388, in forward
    y_cpu = y.detach().cpu().numpy()
            ^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

```

Il problema non si verifica ogni volta (una volta si e una circa e quando non si verifica questo errore se ne verifica un'altro dovuto al gradiente ma non credo siano correlati).

Cercando in internet ho trovato che potrebbe essere dovuto alla versione troppo recente di pytorch
così ho effettuato il downgrade alla versione torch==1.7.1+cu101.

### ATTENZIONE
Per poter istallare la versione indicata di torch bisogna effettuare l'installazione anche di una versione precedente di python (io ho utilizzato la 3.7 è cosigliato pertanto di usare un venv).
Le dipendenze installate dopo aver installato la versione corretta di python sono le seguenti e sono state fatte in questo ordine.


```sh
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install matplotlib==3.3.4
pip install nibabel==3.2.1
pip install scipy==1.6.1

pip install thop
pip install timm

pip install opencv-python==3.4.11.45
# Nel caso opencv non scarichi automaticamente numpy
pip install numpy==1.20.1

# Installare prima visual studio code (solo le cose base di c++) 
# altrimenti la seguente libreria potrebbe dare errore
pip install libtiff

# Aggiornare pip come segue altrimenti il pacchetto nnunet non viene scaricato
python -m pip install --upgrade pip==21.0
pip install nnunet==1.6.6

```

La cartella con la versione di python viene installata al seguente path:
``C:\Users\NOMEUTENTE\AppData\Local\Programs\Python``


<br/><br/>

Sfortunatamente il problema persiste.

Maybe this can fix the issue?
https://discuss.pytorch.org/t/assertion-n-idx-dim-0-idx-dim-index-size-index-out-of-bounds-failed/166275


Per sistemare l'errore ho provato differenti approcci e guardando il link sopra indicato ho provato differenti metodi:

Suggerito quanto segue

L'errore che stai riscontrando sembra essere dovuto a un'operazione di indicizzazione non valida durante l'utilizzo del metodo scatter_() su un tensore y. L'errore è specificato come "index out of bounds", il che significa che uno o più valori dell'indice utilizzato per l'indicizzazione sono al di fuori dei limiti validi.

Per risolvere questo problema, segui questi passaggi:

Controlla il messaggio di errore completo, in particolare la traccia dello stack (stack trace) che dovrebbe indicare l'operazione di indicizzazione che ha causato l'errore. Potrebbe esserci un numero di riga specificato nella traccia dello stack che ti aiuterà a identificare la posizione esatta del problema.

Una volta individuata l'operazione di indicizzazione problematica, assicurati che i valori nell'indice (y_.long()) siano all'interno dei limiti validi. Ad esempio, se y_.long() contiene indici che rappresentano le etichette di classe, assicurati che siano compresi tra 0 e il numero totale di classi meno uno.

Se gli indici sono fuori dai limiti, potresti dover controllare come vengono generati o passati all'operazione scatter_(). Assicurati che l'indice sia correttamente calcolato in base ai tuoi requisiti e al formato dei dati.

Se gli indici sembrano corretti, potrebbe esserci un problema con la dimensione del tensore y. Assicurati che y abbia la stessa forma del tensore x, poiché scatter_() richiede che i due tensori abbiano le stesse dimensioni.

Inoltre, verifica che il dispositivo corretto sia utilizzato per eseguire le operazioni. Nel tuo caso, stai controllando se la GPU è disponibile e sposti y sulla GPU. Assicurati che x e y siano sullo stesso dispositivo (CPU o GPU) prima di eseguire scatter_().



Come prima cosa ho applicato quanto segue: 
Ho impostato num_class a 2 per vedere quanti valori erano fuori dal range e non venivano segnati indici errati.

Il valore di num_classes, ovvero il numero totale di classi, dipende dal contesto del tuo problema specifico. È una variabile che dovresti conoscere o definire in base al dataset o al problema che stai affrontando.

Se stai lavorando con un problema di classificazione, num_classes rappresenta il numero di classi uniche presenti nei tuoi dati di addestramento o nel tuo dataset. Ad esempio, se stai eseguendo la classificazione di immagini in 10 diverse categorie, num_classes sarà 10.

```python
num_classes = 2
index_max = torch.max(y_.long())
index_min = torch.min(y_.long())

# Verifica i limiti validi degli indici
valid_index_min = 0  # Il valore minimo dell'indice valido
valid_index_max = num_classes - 1  # Il valore massimo dell'indice valido, dove `num_classes` è il numero totale di classi

if index_min < valid_index_min or index_max > valid_index_max:
    print("Valori dell'indice fuori dai limiti validi!")
    # Aggiungi qui le azioni correttive necessarie
else:
    # Gli indici sono all'interno dei limiti validi
    # Prosegui con il tuo codice

```

Di conseguenza gli indici mi sembravano corretti (o almeno credo). Quindi ho verificato che i tensori x e y abbiano le stesse dimensioni. Questo percè come precedentemente detto l'errore sembrava fosse nel metodo scatter_.

Ho dunque provato con:

```python

# Controlla se le forme dei tensori x e y non corrispondono
if x.shape != y.shape:
    # Adatta la dimensione di y alla forma di x
    y = y.view(x.shape)

```
Implementandolo tra la verifica di cuda e lo scatter. Ma non ha funzionato.

Ho provato ad applicare ``softmax_helper()`` ad y_

```python

y_ = softmax_helper(y)

```

E anche qui purtroppo non ha funzionato.


Allora ho provato andando nel file ``Train.py`` a verificare i dati in input e ho notato che utilizzando direttamente l'immagine al posto dei valori P1... quindi senza applicare la funzione
``model()``Funzionava senza problemi (devo capire ancora il perchè).




Problema successivo è che il tensore non ha la backword che aveva se usavo P1. dunque ho dovuto commentare queste la riga nel file ``Train.py`` riguardante la ``backward()``

```python

 # ---- backward ----
#loss.backward()
clip_gradient(optimizer, opt.clip)
optimizer.step()

```

Successivamente  ho dovuto commentare per un'altro errore dovuto alla mancanza della loss_P4 un'ulteriore riga. E l'ho sostituita con la nuova loss.

    ```python
        from
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P4.data, opt.batchsize)

        to
            # ---- recording loss ----
            if rate == 1:
                #loss_P2_record.update(loss_P4.data, opt.batchsize)
                loss_P2_record.update(loss.data, opt.batchsize) 
    ```

Ora il codice funziona ma da un'altro errore molto più avanti perchè non trova una cartella quindi da verificare quello.

Era un errore in

```python
        from
            test1path = folder_path.MY_TRAIN_FOLDER_PATH

        to
            test1path = folder_path.MY_TEST_FOLDER_PATH
    ```

avrò sbagliato io in precedenza a inserire il percorso.


RuntimeError: The size of tensor a (20) must match the size of tensor b (19) at non-singleton dimension 3

torch.Size([2, 1, 128, 128]) torch.Size([2, 1, 128, 128]) torch.Size([2, 1, 128, 128]) torch.Size([2, 1, 128, 128])





for epoch in range(1, opt.epoch + 1):











<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

## Messaggio di warning da sistemare (non da problemi nel codice al momento)

\\path\HSNet\PyTorch\env\Lib\site-packages\torch\nn\_reduction.py:42: 
            UserWarning: size_average and reduce args will be deprecated, 
            please use reduction='mean' instead.
            warnings.warn(warning.format(ret))