from surprisal import AutoHuggingFaceModel
#from scipy.io import savemat
#import numpy
import json

class SurprisalDict:

    def __init__(self):
        surprDict = {} # Subjectrun (e.g., subj-01_run1) as key. Ondsdurdict as value.
        modfile = 'modalities.csv'
        modfile = self.clean_modality_file(modfile)
        self.m = AutoHuggingFaceModel.from_pretrained('Cedille/fr-boris', model_class = 'gpt')
        surp_dict = self.get_surprisal(modfile)

        with open("surpdict.json", "w") as sd:
            json.dump(surp_dict, sd)

    def clean_modality_file(self, modfile):
        # returns a dict where each key is the subjrun (e.g., subj-25_run-4) and values is a list with lines. Each line contains info about an utterance
        with open(modfile) as mf:
            mf = mf.readlines()[1:]
            subjrun_dict = {}
            for line in mf:
                if len(line) == 1:
                    continue
                else: 
                    line = line.strip('\n').strip('$').strip('*').split(',')
                    subjrun, cond, onset, duration, prod, n_tok, utterance = \
                        line[0] + '_run-' + line[2], line[1], line[4], line[5], line[6], line[8], line[9]
                    utterance_info = tuple([cond, onset, duration, prod, n_tok, utterance])
                    subjrun_dict.setdefault(subjrun, []).append(utterance_info)
        return subjrun_dict

    def get_surprisal(self, modfile):
        # Returns a dict where subjrun is key and value is a dict where each key is a line with info about one utterance and value is the surprisal of the words in that utterance. 
        surp_dict = {}
        for subjrun in modfile:
            surp_dict[subjrun] = {}

            for line in modfile[subjrun]:
                utterance = line[5]
                
                if line[0] == 'human' and len(utterance) > 0:
                    line = ','.join(line)
                    surprisal_data = json.dumps(self.m.surprise(utterance), indent=4, sort_keys=True, default=str).strip('[').strip(']').split()[1:-1]
                    surprisal_data = surprisal_data[len(surprisal_data)//2+1:]
                    surp_dict[subjrun][line] = surprisal_data
                else: 
                    line = ','.join(line)
                    surp_dict[subjrun][line] = None

        return surp_dict

c = SurprisalDict()
c