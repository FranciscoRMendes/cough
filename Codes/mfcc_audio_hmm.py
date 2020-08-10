df = pd.read_csv('cSpace_file_diagnosis.csv')
patients_female = df[df.Sex=='Female'].File.to_list()
patients_male = df[df.Sex=='Male'].File.to_list()

patients = df.First_last.unique()
data_dir = 'wav/'
len(patients_female), len(patients_male)

patients

p = [x for x in patients_male if 'Joseph_F' in x]

patient_cough = []
for i in p:
    y,fs = librosa.load(data_dir+i, sr=16000)
    b,a = signal.butter(N=8, Wn=1000/8000, btype='lowpass')
    y = signal.filtfilt(b,a,y)
    y = np.asfortranarray(y)
    print(i, end=' | ')
    mfcc = librosa.feature.mfcc(y, sr=fs, n_mfcc=20)
    patient_cough.append(mfcc.T)

lengths = [len(x) for x in patient_cough]
X = np.concatenate(patient_cough)
X.shape, lengths

n_components = 3
patient_model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=100,
                                )
patient_model.transmat_ = np.zeros((n_components,n_components)) + 1/n_components

patient_model.fit(X = X, lengths=lengths)
np.round(patient_model.transmat_, 3)

gn, gn_state = patient_model.sample(300)
y_gen = librosa.feature.inverse.mfcc_to_audio(mfcc = gn.T)
