
### Objectif: Pour une musique donnée, ressortir une chorégraphie générée par une IA

![[Pasted image 20230702222939.png]]
											*Pipeline du projet*

## Tâches à faire

### 1) Classifier les musiques avec un algorithme d'IA

Cette tâche consiste à faire la classification des différentes musiques en fonction des différents genres (hip hop, dance afro, danse classique). Cette classification va permettre de choisir les pas de danse en fonction du genre de la musique.

#### a) Benchmark des base de données à utiliser

Le Benchmark est dans un fichier excel avec les quelques caractérisques des données. Parmi les données trouvés, il se trouve que l'ensemble GTZAN comporte les meilleures données (bien structurées) pour faire une classification de musique. 

<u><b>Problèmes rencontrés</b></u>: 

Pour la classification des musiques, on s'est basé sur 3 genres (afro, hip hop et classique). Mais, les données de musiques Afro sont manquantes d'où la nécessité de faire une collecte de données. Les données collectées seront ensuite mises sous la forme des données GTZAN en utilisant des logiciels.

<u><b>Caractéristiques des données GTZAN</u></b>

**genres original :** Une collection de 10 genres avec 100 audio chacun. Chaque audio a une durée de 30 secondes.

**images original :** Représentation visuelle pour chaque audio. Les réseaux neuronaux constituent l'un des moyens de classer les données. Étant donné que les réseaux neuronaux (comme le CNN, que nous utiliserons aujourd'hui) prennent généralement en charge une sorte de représentation d'image, les fichiers audio ont été convertis en spectrogrammes Mel pour rendre cela possible.

**CSV file :** Caractéristiques du fichier audio. Colonnes du fichier csv: filename,length,chroma_stft_mean,chroma_stft_var,rms_mean,rms_var,spectral_centroid_mean,spectral_centroid_var,spectral_bandwidth_mean,spectral_bandwidth_var,rolloff_mean,rolloff_var,zero_crossing_rate_mean,zero_crossing_rate_var,harmony_mean,harmony_var,perceptr_mean,perceptr_var,tempo,mfcc1_mean,mfcc1_var,mfcc2_mean,mfcc2_var,mfcc3_mean,mfcc3_var,mfcc4_mean,mfcc4_var,mfcc5_mean,mfcc5_var,mfcc6_mean,mfcc6_var,mfcc7_mean,mfcc7_var,mfcc8_mean,mfcc8_var,mfcc9_mean,mfcc9_var,mfcc10_mean,mfcc10_var,mfcc11_mean,mfcc11_var,mfcc12_mean,mfcc12_var,mfcc13_mean,mfcc13_var,mfcc14_mean,mfcc14_var,mfcc15_mean,mfcc15_var,mfcc16_mean,mfcc16_var,mfcc17_mean,mfcc17_var,mfcc18_mean,mfcc18_var,mfcc19_mean,mfcc19_var,mfcc20_mean,mfcc20_var,label

<u><b>Récupération des données et ses caractéristiques</u></b>

**1) Collecte des données**

Les données manquantes pour compléter notre ensemble sont les données des musiques afro. J'ai donc collecté les musiques afro puis je les ai segmentés en 30 sec chacun  après les avoir convertir en fichier .wav sur une plateforme en ligne. Afin de ressortir les caractéristiques de chaque audio, j'ai fait un script python utilisant la bibliothèque **librosa**.

***Decoupage du fichier audio***
```python
import librosa
import math
import soundfile as sf

# Load audio file
audio_file = 'Afrobeat_wavFile\\afrobeat_abracadabra.wav'
y, sr = librosa.load(audio_file)
filename = "afrobeat_abracadabra"
# Get duration of audio in seconds
duration = librosa.get_duration(y=y, sr=sr)
# Calculate number of 30-second segments
num_segments = math.ceil(duration / 30)
# Divide audio into 30-second segments
for i in range(num_segments):
    segment_start = i * 30 * sr
    segment_end = min(segment_start + 30 * sr, len(y))
    segment = y[segment_start:segment_end]
    # Save segment to file
    segment_filename = 'segment{}_{}.wav'.format(i+1, filename)
    sf.write(segment_filename, segment, sr)
```

***Extraction des caractéristiques avec librosa***

```python
import librosa
import numpy as np
import os

# Load existing features from features_30_sec.csv
existing_features = {}
with open('features_30_sec.csv', 'r') as f:
    header = f.readline().strip().split(',')
    for line in f:
        values = line.strip().split(',')
        filename = values[0]
        label = values[-1]
        existing_features[filename] = label
  

# Collect feature vectors for all audio files in the folder
feature_vectors = []
for audio_file in os.listdir('genres_original\\afrobeat'):
    if not audio_file.endswith('.wav'):
        continue
    audio_path = os.path.join('genres_original\\afrobeat', audio_file)
  
    # Load audio file
    y, sr = librosa.load(audio_path)
  
    # Calculate features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    harmony, perceptr = librosa.effects.hpss(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # Calculate feature means and variances
    feature_means = [np.mean(chroma_stft), np.var(chroma_stft), np.mean(rms), np.var(rms),
                     np.mean(spectral_centroid), np.var(spectral_centroid), np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
                     np.mean(rolloff), np.var(rolloff), np.mean(zero_crossing_rate), np.var(zero_crossing_rate),
                     np.mean(harmony), np.var(harmony), np.mean(perceptr), np.var(perceptr),
                     tempo]

    for i in range(mfcc.shape[0]):
        feature_means.extend([np.mean(mfcc[i]), np.var(mfcc[i])])

    # Add filename, length, and label to feature vector
    feature_vector = [audio_file, librosa.get_duration(y=y, sr=sr)]
    feature_vector.extend(feature_means)
    label = existing_features.get(audio_file, 'afrobeat')
    feature_vector.append(label)
    feature_vectors.append(feature_vector)

# Append new feature vectors to features_30_sec.csv
with open('features_30_sec.csv', 'a') as f:
    for feature_vector in feature_vectors:
        f.write(','.join(map(str, feature_vector)) + '\n')
```

Une fois, le fichier les caractéristiques des audio de 30 sec obtenus , le meme processus a été fait pour le fichier des caractéristiques des audio de 3 secondes. 


Un fois les données obtenues, il faut ressortir une représentation visuel des audio avec la bibliothèque librosa.
Nous allons representer un spectogramme  qui représente la variation de la fréquence en fonction du temps. Il permet de visualiser les composantes fréquentielles du signal audio. 

````python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os


# Set the input audio directory path
audio_dir = 'D:\Tempory Dowloads\\archive\Data\genres_original\\afrobeat'

# Loop over all audio files in the directory
for audio_file in os.listdir(audio_dir):
    if audio_file.endswith('.wav'):
        # Load audio file
        audio_path = os.path.join(audio_dir, audio_file)
        y, sr = librosa.load(audio_path)

        # Generate spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, y_axis='linear', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        # Create the "images" directory if it doesn't exist
        # if not os.path.exists('images'):
        #     os.makedirs('images')

  

        # Save the spectrogram as a PNG file

        image_file = os.path.splitext(audio_file)[0] + '.png'
        image_path = os.path.join('D:\Tempory Dowloads\\archive\Data\images_original\\afrobeat', image_file)
        plt.savefig(image_path, dpi=300, bbox_inches='tight')


        # Close the plot
        plt.close()
````

Music genre classification: https://www.projectpro.io/article/music-genre-classification-project-python-code/566



#### b) Elaboration du modèle à utiliser

Pour la classification des genres audio, nous avons utilisé l' [ensemble de données de classification des genres GTZAN](https://www.kaggle.com/esratmaria/music-genre-recognition) qui compte environ 100 pistes chacune de taille et de gamme de fréquences similaires. Ils incluent les 11 genres musicaux suivants - Afrobeat, Blues, Classique, Country, Disco, Hip Hop, Jazz, Métal, Pop, Reggae et Rock.

Nous avons deux approches pour cette classification

- Classification d'images sur les spectogrammes et ondelettes générés
- Classification à l'aide des fonctionnalités extraites des formes d'onde audio

<u><b>Sous-approche I : Classification d'images sur les spectogrammes et ondelettes générés</b></u>

Pour chaque fichier audio, des images de spectrogrammes ont été générés. Après le prétraitement des données et les fractionnements train test, nous pouvons alimenter les données d'apprentissage dans les 3 modèles suivants.

- **Réseau de neurones convolutifs (utilisant des spectrogrammes)**
- **Apprentissage par transfert (à l'aide de spectrogrammes)**
- **Formation multimodale (utilisant à la fois des spectrogrammes et des ondelettes)** : Nos données ne se composent pas des ondelettes, donc nous allons explorer les deux premières méthodes.

![[Pasted image 20230710191105.png]]
			<i>Flux de processus de classification d'images</i>

<u><b> a) Définition spectogrammes et ondelettes</b></u>

![[Pasted image 20230710191354.png]]
			<i>Spectogrammes de différents genres</i>

 **Un spectrogramme** est une représentation visuelle d'une chanson, avec un axe indiquant la fréquence et l'autre indiquant le temps, et les niveaux de volume étant représentés par différentes couleurs.  Ces images sont produites à l'aide de la bibliothèque **librosa** en python.

![[Pasted image 20230710192553.png]]
<i>Ondelettes de différents genres</i>

Les ondelettes sont des représentations visuelles des propriétés spectrales et temporelles des signaux audio. Nous pouvons à nouveau utiliser la bibliothèque **Librosa** pour les générer en python.

D'apès cet article (https://medium.com/@aritrachowdhury95/music-genre-classification-using-deep-learning-audio-and-video-770173980104), les résultats obtenus sont de 72,25% pour le modèle CNN,  56% pour le modèle MobileNet utilisant un transfert learning et 52,25% pour le modèles utilisant les spectogramme et les ondelettes.

<u><b>Sous-approche II : Classification d'images sur les spectogrammes et ondelettes générés</b></u>

Les formes d'onde audio se composent de plusieurs fonctionnalités. Suite à cet [article](https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8) , 5 caractéristiques pertinentes du signal audio ont été extraites pour cette approche.

- [**Zero-Crossing Rate**](https://en.wikipedia.org/wiki/Zero-crossing_rate) **:** C'est la vitesse à laquelle un signal passe de positif à zéro à négatif ou de négatif à zéro à positif. Intuitivement, il a généralement des valeurs plus élevées pour les sons très percussifs comme ceux du métal et du rock.
- [**Centroïde spectral**](https://en.wikipedia.org/wiki/Spectral_centroid#:~:text=The%20spectral%20centroid%20is%20a,of%20brightness%20of%20a%20sound.) **:** Il a une connexion robuste avec l'impression de luminosité d'un son et indique où se situe le « centre de masse » d'un son.
- [**Spectral Roll-off**](https://librosa.org/doc/main/generated/librosa.feature.spectral_rolloff.html) **:** Il représente la fréquence en dessous de laquelle se trouve un pourcentage spécifié de l'énergie spectrale totale.
- [**Chroma Frequencies**](https://en.wikipedia.org/wiki/Chroma_feature) **:** Il s'agit d'une représentation de l'audio musical dans laquelle le spectre entier est projeté sur 12 cases représentant les 12 demi-tons distincts (ou chroma) de l'octave musicale.
- - [**Coefficients cepstraux de fréquence Mel (MFCC)**](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum#:~:text=Mel%2Dfrequency%20cepstral%20coefficients%20(MFCCs,%2Da%2Dspectrum%22).&text=MFCCs%20are%20commonly%20derived%20as,windowed%20excerpt%20of)%20a%20signal.) **:** Ils sont une représentation du [spectre de puissance](https://en.wikipedia.org/wiki/Power_spectrum) à court terme d'un son, basée sur une [transformée en cosinus linéaire](https://en.wikipedia.org/wiki/Cosine_transform) d'un [spectre de puissance logarithmique](https://en.wikipedia.org/wiki/Power_spectrum) sur une [échelle de fréquence mel](https://en.wikipedia.org/wiki/Mel_scale) [non linéaire](https://en.wikipedia.org/wiki/Nonlinear_system) . En termes simples, ils sont une représentation de la façon dont l'oreille humaine perçoit le son.

![[Pasted image 20230710193318.png]]
<u><b>(de gauche à droite, de haut en bas) Zero-Crossing Rate, Spectral Centroid, Spectral Roll-off, Chroma Frequencies, Mel-Frequency Cepstral Coefficients ( Lien de [référence](https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8) )</b></u>

Après l'extraction des caractéristiques, plusieurs algorithmes de classification multi classes ont été appliqués. Les résultats obtenus sont comme suit.

- **Le réseau neuronal séquentiel** atteint une **précision de test parmi les 2 meilleures de 47,0 %**
- **Random Forest** atteint une **précision de test Top-2 de 70,4 %**
- **Les machines à vecteurs de support avec noyau polynomial** atteignent une **précision de test parmi les 2 meilleures de 45,6 %**
- **La régression logistique** atteint une **précision de test Top-2 de 50,0 %**


<u><b>Conclusion:</b></u>
Je vais opter pour l'utilisation des modèles **CNN**

J'ai lancé mon modèle CNN sur mes fichiers CSV mais le problème qui s'est posé c'est que notre modèle doit prédire sur les fichiers audio et non les données écrites. D'où la nécessité de concevoir un modèle Pytorch pour les entrainer sur des audio.

<u><b>Conception du modèle Pytorch</u></b>
Pour la classification du modèle de classification d'audio, j'ai utilisé le modèle à base de Pytorch. De plus, afin d'affiner mon modèle les genres de musiques qui ne m'intéressent pas ont été supprimé. Donc, il ne me reste plus que les musiques afrobeat , classique et hip hop. 

***Problèmes rencontrés***
Le modèle a été entrainé sur des audio de 30s donc pour que le modèle puisse prédire de bons résultats, il faut lui passer des audio de 30s également. On va d'abord faire avec ce modèle.

Accuracy du modèle est : 93.33%

![[Pasted image 20230719183616.png]]

Source of code: https://deeplearning.neuromatch.io/projects/ComputerVision/spectrogram_analysis.html

### 2) Classification des pas de danse

Cette tâche consiste à faire une classification de pas de danse en fonction du genre de la musique.

La ***première étape*** ici consiste à collecter les données  de différents pas de danse étiquetés afin de les classifier par genre(hip hop, afro, salsa, danse classique). 

<u><b>Collecte des données</u></b>

Le but ici est d'avoir les images des pas de danse pour pouvoir le faire tourner sur nos modèles(estimatin de pose, cnn, ...). Nous avons donc ces différentes étapes:
- Collecte des videos de danse pour chaque type de danse (afro, hip hop, classique)
- Découper les videos en petites partitions pour representer chaque pas de danse
- recupérer les images clés de chaque partitions de video
- Encoder les étiquettes sous forme numériques (à voir)
***Structure des données***
afrobeat -> images de pas de danse pour chaque partition de données

La ***deuxième étape*** consiste à former un modèle pour classifier les pas de danse.

<u><b>Etapes de développement du modèle</u></b>

- **Division des données en données de train et de test pour chaque genre**
- **Prétraitement des données**
      Pour le prétraitement des données, j'ai utilisé la fonction ***custom_transform()*** pour transformer les données. Cette fonction a plusieurs fonctionnalités telles que: 
	- transforms.Grayscale transforme l'image en nuances de gris.
	    
	- transforms.Pad ajoute un padding sur l'image, c'est-à-dire qu'il remplit les bords avec un pixel de valeur 0. Ici le padding est défini dans le tuple padding.
	    
	- transforms.Resize redimensionne l'image à une taille de 90x160 pixels.
	    
	- transforms.RandomHorizontalFlip flippe horizontalement l'image aléatoirement, pour augmenter les données.
	    
	- transforms.ToTensor transforme l'image PIL/numpy en un tenseur torch.
	    
	- transforms.Normalize normalise les pixels de l'image entre -1 et 1, en soustrayant la moyenne et en divisant par l'écart type. Ici on normalise par rapport à la moyenne 0.5 et l'écart type 0.5.

	 Le code est le suivant :
	````python
	def custom_transform(padding=(0,0)):
	    """
	    padding[0] is the height
	    padding[1] is the width
	    """
	    custom = transforms.Compose([
	                        transforms.Grayscale(num_output_channels=1),
	                        transforms.Pad(padding, fill=0),
	                        transforms.Resize((90, 160)),
	                        transforms.RandomHorizontalFlip(),
	                        transforms.ToTensor(),
	                        transforms.Normalize([0.5],
	                                            [0.5])])
	    return custom
	````
	Il y a également la fonction ***collate_function()*** pour redimensionner les images qui n'ont pas la taille 16:9
  - **Conception du modèle**
       It is a convolutionneuronnal network(CNN) with this architecture.
       1. It is a Convolutional Neural Network (CNN) for image classification.
    
	2. It takes grayscale images of size 1x90x160 as input. This is based on the input channel size of 1 in the first conv layer.
	    
	3. It has 4 convolutional layers with kernel sizes of 3x3 and strides of 2. This helps extract features from the images at different scales.
	    
	4. It uses BatchNorm and Dropout after each conv layer to reduce overfitting.
	    
	5. It flattens the output of the conv layers and feeds it to a linear (dense) layer.
	    
	6. The linear layer has 512, 256 and 128 nodes with Dropout in between. This helps reduce overfitting.
	    
	7. The final linear layer has 3 nodes, indicating this model is intended for a 3-class classification problem.
	    
	8. The model is defined to work on both CPU and GPU using the 'device' variable.
	    
	9. The model is first defined, then sent to the correct device (CPU or GPU) using .to(device).
	
	10. We have this parameters for the model: 
		-  epochs : Le nombre d'époques d'entraînement, ici défini à 10.
		    
		- batch_size : La taille des batchs, soit 32 échantillons d'entraînement par itération.
		    
		- learning_rate : Le taux d'apprentissage du modèle, ici 0.0005.
		    
		- print_every : La fréquence d'affichage des résultats, ici toutes les 40 itérations.
		    
		- device : Le périphérique d'exécution, GPU ou CPU.
		    
		- train_loader, test_loader : Les chargeurs des jeux de données d'entraînement et de test.
		    
		- classes : La liste des classes de classification, ici 'Afrobeat', 'Classical' et 'hiphop'.
		    
		- model : Le modèle CNN défini précédemment.
		    
		- optimizer : L'optimiseur Adam, avec un taux d'apprentissage de learning_rate.
		    
		- scheduler : Le scheduler qui décrémente le taux d'apprentissage au fil des époques.
		    
		- criterion : Le critère de perte, ici une CrossEntropyLoss pour la classification.
  - **Résultats du modèles**
	  ![[Pasted image 20230726141823.png]]
	  ![[Pasted image 20230726141833.png]]
	  
- **Test du modèle**
	 Pour tester mon modèle, il y a plusieurs étapes à faire avoir un résultat :
	 - Récupération d'une vidéo
	 - Transformation de cette vidéo en images comme pour la collecte de données
	 - Prétraitement des images 
	 - Envoie de ces images au modèle
![[Pasted image 20230808165141.png]]

Source: https://github.com/Yuning-J/Transpondancer/blob/main/src/Ballet/train.py
### 3) Première démo

Faire un assemblage des premières fonctionnalités. On donne une musique à l'algorithme  et il classifie la musique. Une fois la musique classifié, l'algorithme recherche les pas de danse approprié en fonction du genre de la musique ressorti.

### 4) Lier les pas de danse à la musique

Pour chaque frame de la musique, lier les pas de danse approprié en fonction du genre .
 Le modèle FACT a été utilisé pour faire un generateur de danse pour AIST++
### 5) Associer les pas de danses à la musique


Source: https://datasets.activeloop.ai/docs/ml/datasets/gtzan-genre-dataset/#:~:text=The%20GTZAN%20genre%20collection%20dataset%20consists%20of%201000%20audio%20files,that%20represent%2010%20music%20genres.