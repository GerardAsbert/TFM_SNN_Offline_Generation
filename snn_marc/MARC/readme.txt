El meu codi per generar varacions en versio simple en memoria (per simplicitat d'outputs si el voleu correr i exectuar despres), no cal fer moltes iteracions ja que despres suavitzes.



El que s'ha de fer:
- Al fitxer de handwritting canvies el nom del fitxer .txt que poses com a input (ha d'estar a la mateixa carpeta que el notebook o has de canviar el path)
- Retoques el que vulguis de l'arquitectura (neurones, iteracions, etc.) Jo ho deixo a poques perque despres ho suavitzem
- Toques el jitter que esta al costat del loop d'entrenament (jo ho deixaria a uns 0.75 ms)

Un cop ho tens vas a execute models.
- L'ouput del primer notebook haura deixat tant un .pkl amb els pesos com un numero que és el que tarda en executar una iteració. Aquestes dues coses les heu de modificar a execute_models (simplement copiar el nom del fitxer amb el path)
- Assegurar que el nombre de neurones de l'arquitectura de la run d'execute models es la mateixa que la que heu entrenat. 
- Jugar amb el jitter de sortida i els valors de suavitzat (els que estan posats us funcionaran be). Cada cop que executeu l'arquitectura (es una iteracio i sense aprenentatge així que és automatica) us sortiran lletres diferents. Feu-ne tantes com volgueu.
- Quan volgueu executar lletres haureu d'escriure a dalt el text que volgueu escriure a un input que trobareu a les primeres cel·les. La "lletra" que es sera la mateixa que estigui configurada al notebook de training (per simbols jo simplement faig servir la lletra "a" ja que no hi ha cap sostingut al codi ASCII)

Jo copio directament l'output a clipboard, pero podeu escriure un petit script que us guardi l'imatge al path que voleu.
