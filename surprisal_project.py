import torch
from surprisal import AutoHuggingFaceModel
from scipy.io import savemat
import numpy
import json

m = AutoHuggingFaceModel.from_pretrained('Cedille/fr-boris', model_class = 'gpt')


zero_freq = [] #för att kika på vilka ord/typer av ord som inte förekommer i frekvenstabellen


###listor för slutliga dict-values
word_surprisals_comp_h = []
word_surprisals_comp_r = []
word_surprisals_prod_h = []
word_surprisals_prod_r = []
word_frequency_comp_h = []
word_frequency_comp_r = []
word_frequency_prod_h = []
word_frequency_prod_r = []
word_onsets_comp_h = []
word_onsets_comp_r = []
word_onsets_prod_h = []
word_onsets_prod_r = []
word_durations_comp_h = []
word_durations_comp_r = []
word_durations_prod_h = []
word_durations_prod_r = []

silence_onsets_h = []
silence_onsets_r = []
silence_durations_h = []
silence_durations_r = []

isi_onsets = []
isi_durations = []
pres_onsets = []
pres_durations = []


### Returnerar surprisal ackumulerad per ord
def temp_stack(temp_list):
	res = {"word":'', "surprisal":0}
	for i in temp_list:
		res["word"] += i[0]
		res["surprisal"] += float(i[1])
	return(res["surprisal"])


### Returnerar ons och durs för ISI och PRES från logfiles samt för tystnader från transitionsfilen och justerad via conv_onsets
def get_silence_ISI_PRES_ons_durs(conv_onsets, subjrun):
	silence_durs_temp_h = []
	silence_durs_temp_r = []
	silence_ons_temp_h = []
	silence_ons_temp_r = []
	with open('transitions.csv') as trans_f:
		rows = trans_f.readlines()[1:]
		for row in rows:
			currentRow = row.split(',')
#			print(subjrun[:2], subjrun[-1], currentRow[0][-2:], currentRow[2])

			if currentRow[1] == 'human' and currentRow[0][-2:] == subjrun[:2] and currentRow[2] == subjrun[-1]:
				silence_durs_temp_h.append(float(currentRow[5]))
				silence_ons_temp_h.append(float(currentRow[4]) + float(conv_onsets[int(currentLine[3]) -1]))

			if currentRow[1] == 'robot' and currentRow[0][-2:] == subjrun[:2] and currentRow[2] == subjrun[-1]:
				silence_durs_temp_r.append(float(currentRow[5]))
				silence_ons_temp_r.append(float(currentRow[4]) + float(conv_onsets[int(currentLine[3]) -1]))

	presentation_onsets = []
	presentation_durations = []
	fixation_onsets = []
	fixation_durations = []
	with open('logfiles/sub-' + subjrun[:2] + '_task-convers_run-0' + subjrun[-1] + '_events.tsv') as log_f:
		for log_l in log_f:
			log_l = log_l.split('	')
			print('log_l (as list) get ISI PRES: ', log_l)
			if log_l[2] == 'ISI':
				fixation_onsets.append(float(log_l[0]))
				fixation_durations.append(float(log_l[1]))
			if log_l[2] == 'INSTR1':
				presentation_onsets.append(float(log_l[0]))
				presentation_durations.append(float(log_l[1]))

#	print('silence durs, ons listlengths: ', len(silence_durs_temp_h), len(silence_durs_temp_r), len(silence_ons_temp_h), len(silence_ons_temp_r))
#	print('listlängder isi och fixation: ', len(fixation_onsets), len(fixation_durations), len(presentation_onsets), len(presentation_durations))

	return(silence_ons_temp_h, silence_ons_temp_r, silence_durs_temp_h, silence_durs_temp_r, \
	fixation_durations, fixation_onsets, presentation_durations, presentation_onsets)


### Returnerar en dict med names, ons, durs, freqs, surps för varje currentLine ackumulerad per subjrun, returnerar också conv-onsets
def get_surprisals_freqs_durs_ons():
	print('subjrun_line: ', subjrun_line)

	word_onsets = []
	word_durations = []
	word_surprisal = []
	word_frequency = []

	vocalizations = ['euh', 'mh', '***', '@', '*'] #lägg till andra vocalizations?
	utterance_words = subjrun_line[9].split(' ')
	for wo in utterance_words:
		utterance_words[utterance_words.index(wo)] =  wo.strip(' ')
		if wo.startswith('$'):
			utterance_words[utterance_words.index(wo)] =  wo.strip('$')
		if wo in vocalizations:
			utterance_words.remove(wo)
	n_words = len(utterance_words)
#	print('n_words: ', n_words)
	utterance_words_rejoined = ' '.join(utterance_words)
#	print('utterance_words: ', utterance_words)

	### tar fram och ackumulerar surprisals fr vardera ord i ett yttrande
	temp_list = []
	list_of_tuples = []
	utterance_surprisal = m.surprise(utterance_words_rejoined)
	utterance_surprisal = list(utterance_surprisal)
	for utterance_tuples in utterance_surprisal:
		utterance_tuples = tuple(utterance_tuples)
		for t_tuple in utterance_tuples:
			list_of_tuples.append(t_tuple)
		for i, token_tuple in enumerate(list_of_tuples):
			if i == len(list_of_tuples) - 1 or i < len(list_of_tuples) - 1 and list_of_tuples[i + 1][0].startswith('Ġ'):
				temp_list.append(token_tuple)
				word_surprisal.append(temp_stack(temp_list))
				temp_list = []
			else:
				temp_list.append(token_tuple)

	### tar fram konversationernas onsets (för att addera till onset för vardera ord)
	with open('logfiles/sub-' + subj + '_task-convers_run-0' + run + '_events.tsv') as logfile:
		conv_onsets = []
		for log_line in logfile:
			log_line = log_line.split('	')
			if log_line[2] == 'CONV1' or log_line[2] == 'CONV2':
				conv_onsets.append(log_line[0])

#		print('conv_onset: ', conv_onsets)

	#### lägger till floa onset för current conv och därefter vardera ackumulera approximerad ord-onset
	current_onset = float(subjrun_line[4]) + float(conv_onsets[int(subjrun_line[3]) -1])
	mean_duration = float(subjrun_line[5])/int(subjrun_line[8])
	for i, item in enumerate(utterance_words):
		word_durations.append(mean_duration)
		word_onsets.append(current_onset)
		current_onset += mean_duration
#	print('ackumulerad current_onset efetr varje yttrande: ', word_onsets, 'antal ackumulerade onsets: ', len(word_onsets))


	#### matchar frekv från frekvenslista för vardera ord (baserat på ordindelningar fr vardera yttrande)
	with open('1gms/vocab_cs') as frequency_f:
		frequency_f  = frequency_f.readlines()
#		print('längd av freklistan: ', len(frequency_f))
		for w in utterance_words: #utterance_words är en rensad tokeniserad lista av yttrandets ord där vocalizations plockats bort
			for ind, freq_line in enumerate(frequency_f):
				freq_line = freq_line.split('	')
				freq_line[0] = freq_line[0].strip()
				freq_line[0] = freq_line[0].strip('$')
#				print(int(freq_line[1].strip('\\n')))
				if w == freq_line[0]:
					word_frequency.append(float(numpy.log(int(freq_line[1].strip('\\n')))))
#					numpy.log(int(freq_line[1].strip('\\n'))))
					break
				elif ind == len(frequency_f) - 1:
					word_frequency.append(float(0))
#					print('ord som inte finns i listan? /inte fått manuellt tilldelat värde', w)

					#### #om man vill kika på vilka ord som har sk noll-frekvens
#					zero_freq.append(w)
#		print(zero_freq)
#

#	print('listlängder: ', len(word_durations), len(word_onsets), len(word_frequency), len(word_surprisal))


	mdic = {'names':[['ISI'], ['PRES'], ['comp_h'], ['comp_r'], ['prod_h'], ['prod_r'], ['silence_h'], ['silence_r']], \
	'durations': [isi_durations, pres_durations, word_durations_comp_h, word_durations_comp_r, word_durations_prod_h, word_durations_prod_r, \
	silence_durations_h, silence_durations_r], \
	'onsets': [isi_onsets, pres_onsets, word_onsets_comp_h, word_onsets_comp_r, word_onsets_prod_h, word_onsets_prod_r, silence_onsets_h, silence_onsets_r], \
	'pmod' : {'comp_h' : [word_frequency_comp_h, word_surprisals_comp_h], 'comp_r' : [word_frequency_comp_r, word_surprisals_comp_r], \
	'prod_h' : [word_frequency_prod_h, word_surprisals_prod_h], 'prod_r' : [word_frequency_prod_r, word_surprisals_prod_r]}}


	if subjrun_line[1] == 'human' and subjrun_line[7] == '1' and subjrun_line[6] == '0':

		mdic['durations'][2].extend(word_durations)
		mdic['onsets'][2].extend(word_onsets)

		mdic['pmod']['comp_h'][0].extend(word_frequency)
		mdic['pmod']['comp_h'][1].extend(word_surprisal)


	elif subjrun_line[1] == 'robot' and subjrun_line[7] == '1' and subjrun_line[6] == '0':

		mdic['durations'][3].extend(word_durations)
		mdic['onsets'][3].extend(word_onsets)

		mdic['pmod']['comp_r'][0].extend(word_frequency)
		mdic['pmod']['comp_r'][1].extend(word_surprisal)

	elif subjrun_line[1] == 'human' and subjrun_line[7] == '0' and subjrun_line[6] == '1':

		mdic['durations'][4].extend(word_durations)
		mdic['onsets'][4].extend(word_onsets)

		mdic['pmod']['prod_h'][0].extend(word_frequency)
		mdic['pmod']['prod_h'][1].extend(word_surprisal)

	elif subjrun_line[1] == 'robot' and subjrun_line[7] == '0' and subjrun_line[6] == '1':

		mdic['durations'][5].extend(word_durations)
		mdic['onsets'][5].extend(word_onsets)

		mdic['pmod']['prod_r'][0].extend(word_frequency)
		mdic['pmod']['prod_r'][1].extend(word_surprisal)


	return(mdic, conv_onsets)




with open('modalities.csv') as f:


	f = f.readlines()[1:]
	subjrun_dict = {}
	vocalizations = ['euh', 'mh', '***', '@', '*']
	for line in f:
		if line != '' and line != '\n':
			currentLine = line.split(',')

			### LIte grejer för att testa pp mindre data
#			if currentLine[0] in ['subj-01', 'subj-02'] and currentLine[1] == 'human' and currentLine[2] == '3' and currentLine[3] == '6':
#			if currentLine[1] == 'human' and currentLine[2] == '3' and currentLine[3] == '6': #bara för å minska testdata

			run = currentLine[2]
			subj = currentLine[0][-2:]
			if subj + run in subjrun_dict:
				subjrun_dict[subj + run].append(currentLine)
			else:
				subjrun_dict[subj + run] = []
				subjrun_dict[subj + run].append(currentLine)

#			print(currentLine)
#	print(subjrun_dict.keys())

	for subjrun_utterance_list in subjrun_dict.values(): #för varje samling (lista)av yttrande(sträng m kolumner) per subjrun
#		print('utterance_list: ', subjrun_utterance_list)
		for subjrun_line in subjrun_utterance_list:
#			print('subj: ', subjrun_line[0][-2:])

			#### THe follwing two lines to remove one-word utterances and short utterances 
			if len(subjrun_line[9].split(' ')) == 1 or float(subjrun_line[5]) < 0.3:
				continue
			else:
				mdic, conv_onsets  = get_surprisals_freqs_durs_ons()

		subjrun = list(subjrun_dict.keys())[list(subjrun_dict.values()).index(subjrun_utterance_list)]
		print('subjrun: ', subjrun)

		silence_ons_h, silence_ons_r, silence_durs_h, silence_durs_r, fixation_durations, fixation_onsets, \
		presentation_durations, presentation_onsets = get_silence_ISI_PRES_ons_durs(conv_onsets, subjrun)

		mdic['onsets'][6].extend(silence_ons_h)
		mdic['onsets'][7].extend(silence_ons_r)
		mdic['durations'][6].extend(silence_durs_h)
		mdic['durations'][7].extend(silence_durs_r)

		mdic['durations'][0].extend(fixation_durations)
		mdic['onsets'][0].extend(fixation_onsets)

		mdic['durations'][1].extend(presentation_durations)
		mdic['onsets'][1].extend(presentation_onsets)
		print('listlängder mdic(isi och pres): ', len(mdic['onsets'][0]),len(mdic['durations'][0]), len(mdic['onsets'][1]), len(mdic['durations'][1]))


		print('onsets: ', len(mdic['onsets'][0]), len(mdic['onsets'][1]), len(mdic['onsets'][2]), len(mdic['onsets'][3]), \
		len(mdic['onsets'][4]), len(mdic['onsets'][5]), len(mdic['onsets'][6]), len(mdic['onsets'][7]), \
		'durations: ', len(mdic['durations'][0]), len(mdic['durations'][1]), len(mdic['durations'][2]), len(mdic['durations'][3]), \
		len(mdic['durations'][4]), len(mdic['durations'][5]), len(mdic['durations'][6]), len(mdic['durations'][7]), \
		'frequencies: ', len(mdic['pmod']['comp_h'][0]), len(mdic['pmod']['comp_r'][0]), len(mdic['pmod']['prod_h'][0]), len(mdic['pmod']['prod_r'][0]), \
		'suprisals: ', len(mdic['pmod']['comp_h'][1]), len(mdic['pmod']['comp_r'][1]), len(mdic['pmod']['prod_h'][1]), len(mdic['pmod']['prod_r'][1]))

		print(mdic)



		with open("json_dicts_140423/mdic_" + subjrun + ".json", "w") as fp:
			json.dump(mdic, fp)

#		dicname = open("mdic_.json")
#		mdic_ = json.load(dicname)



#		filename = '/home/johanna/surprisal_project/matfiles/ons_durs_freqs_surps_subjrun' + subjrun + '.mat'
#		savemat(filename, mdic)
#		savemat(filename, dic)


		###tömmer listor inför nästa subjrun
		isi_onsets = []
		isi_durations = []
		pres_onsets = []
		pres_durations = []

		word_surprisals_comp_h = []
		word_surprisals_comp_r = []
		word_surprisals_prod_h = []
		word_surprisals_prod_r = []
		word_frequency_comp_h = []
		word_frequency_comp_r = []
		word_frequency_prod_h = []
		word_frequency_prod_r = []
		word_onsets_comp_h = []
		word_onsets_comp_r = []
		word_onsets_prod_h = []
		word_onsets_prod_r = []
		word_durations_comp_h = []
		word_durations_comp_r = []
		word_durations_prod_h = []
		word_durations_prod_r = []

		silence_onsets_h = []
		silence_onsets_r = []
		silence_durations_h = []
		silence_durations_r = []




# ATT GÖRA, OM du vill ha snyggare kod:
# Göra så att logfiles-grejen med onsets inte körs för varje rad utan istället kallas och etablerar variabler för varje subjrun
# Dela upp koden i flera funktioner rent generellt (se tomma rader för ungefärlig indelning)
