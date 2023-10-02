from surprisal import AutoHuggingFaceModel
from scipy.io import savemat
import numpy

class SurprisalDict:

    def __init__(self):
        surprDict = {} # Subjectrun (e.g., subj-01_run1) as key. Ondsdurdict as value.
        modfile = 'modalities.csv'
        modfile = self.clean_modality_file(modfile)
        self.m = AutoHuggingFaceModel.from_pretrained('Cedille/fr-boris', model_class = 'gpt')
        w_surprisal = self.get_surprisal(modfile)
        

        ###listor för slutliga dict-values

    def clean_modality_file(self, modfile):
        # returns a dict where each key is the subjrun (e.g., subj-25_run-4) and values is a list with lines. Each line contains info about an utterance
        with open(modfile) as mf:
            mf = mf.readlines()[1:]
            subjrun_dict = {}
            for line in mf:
                if len(line) == 1:
                    continue
                else: 
                    line = line.strip('\n').split(',')
                    subjrun, cond, onset, duration, prod, n_tok, utterance = \
                        line[0] + '_run-' + line[2], line[1], line[4], line[5], line[6], line[8], line[9]
                    utterance_info = tuple([cond, onset, duration, prod, n_tok, utterance])
                    subjrun_dict.setdefault(subjrun, []).append(utterance_info)
        return subjrun_dict

    def get_surprisal(self, modfile):
        for subjrun in modfile:
            if subjrun == 'subj-25_run-4': #Test with one conversation
                print(subjrun)
                for utterance in modfile[subjrun][5]:
                    surprisal_data = self.m.surprise(utterance)
                    print(surprisal_data)
            else: continue

c = SurprisalDict()
c