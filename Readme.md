## NOTA: DA RISCRIVERE TUTTO IL README
# Pacchetti e Librerie

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
<br/>
<br/>

# Modifiche preliminari

* Modificato il file `pvt.py`. Nella classe `HSNet(nn.Module)` è stato modificato il costruttore `__init__(self, channel=32)` in questo modo:

```python
    from
        path = './pretrained_pth/pvt_v2_b2.pth'
    to
        path = './pre-trained/pvt_v2_b2.pth'
```

<br/>

* Sono stati cambiati i path nel file ``train.py`` per il training ed il test in base a dove sono posizionate le cartelle nel mio computer.
<br/>

* Ridotto il batch size da 8 a 2 nel file ``train.py`` per evitare il seguente errore:

      torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 4.00 GiB total capacity; 3.36 GiB already allocated; 0 bytes free; 3.48 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
      See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
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

# Inizio lavorazioni


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
