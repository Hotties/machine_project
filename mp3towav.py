# pythonbasics.org
from os import path
from pydub import AudioSegment

# files
src = "C:/Users/82105/Downloads/이창민_19011622/1.mp3"
dst = "test.wav"

# convert wav to mp3
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")
