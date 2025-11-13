Para correr el main, en consola: .
`python main.py --wav 'Nombre del wav'` .
El --m indica la compresion, para los frames el tamaño actual es de 512, asi que la compresion del 50% seria poner --m 256 y 30% --m 154 .
Para que se cree el archivo comprimido `python main.py --wav 'Nombre del wav' --save` .
El overlap mientras mas alto mejor se escucha el comprimido pero mayor es el peso del pkl a descomprimir, lo mejor es usar 495 para archivos grandes de audio `python main.py --wav 'Nombre del wav' --overlap 495 --save`.
