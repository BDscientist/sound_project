import wave, struct

waveFile = wave.open('C:\cleanfile\sample.wav','r')

length  = waveFile.getnframes()

print(length)

for i in range(0,length):
    waveData = wave.File.readframes(1)
    data = struct.unpack("<h",waveData)
    print(int(data[0]))


#for i in range(0,length):
