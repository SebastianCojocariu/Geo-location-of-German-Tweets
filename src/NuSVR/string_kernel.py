import multiprocessing as mp
import numpy as np
import re
import time
import os
import math

TRACE_ROWS, TRACE_COLUMNS = None, None

def computeKernel_helper(args):
	row_indexes = args[0]
	column_indexes = args[1]
	row_split = args[2]
	column_split = args[3]
	number_of_p_grams = args[4]
	kernel_type = args[5]

	n = row_indexes[1] - row_indexes[0]
	m = column_indexes[1] - column_indexes[0]
	res = np.zeros((number_of_p_grams, n, m))

	for idx1 in range(row_indexes[0], row_indexes[1]):
		dict_list_1 = row_split[idx1 - row_indexes[0]]
		for idx2 in range(column_indexes[0], column_indexes[1]):
			if idx2 >= idx1:
				dict_list_2 = column_split[idx2 - column_indexes[0]]
				for i in range(number_of_p_grams):
					cnt = 0
					for key in dict_list_2[i]:
						if key in dict_list_1[i]:
							if kernel_type == "presence":
								cnt += 1
							elif kernel_type == "spectrum":
								cnt += dict_list_1[i][key] * dict_list_2[i][key]
							elif kernel_type == "intersection":
								cnt += min(dict_list_1[i][key], dict_list_2[i][key])

					res[i][idx1 - row_indexes[0]][idx2 - column_indexes[0]] = cnt
		if idx1 < TRACE_ROWS and column_indexes[0] < TRACE_COLUMNS:
			print("Completed sample {}/{}".format(idx1, TRACE_ROWS - 1))
	return [row_indexes, column_indexes, res]



def createDictionaries_helper(args):
	corpus = args[0]
	minNgramLen = args[1]
	maxNgramLen = args[2]
	res = []

	for text in corpus:
		dict_list, len_text = [], len(text)
		for d in range(minNgramLen, maxNgramLen + 1):
			aux_dict = {}
			for i in range(len_text):
				if i + d <= len_text:
					substring = text[i: i + d]
					aux_dict[substring] = aux_dict.get(substring, 0) + 1
			dict_list.append(aux_dict)
		res.append(dict_list)
	return res


def computeKernelMatrix(corpus, minNgramLen, maxNgramLen, kernel_type):
	kernel_type = kernel_type.lower()
	if kernel_type not in ["presence", "spectrum", "intersection"]:
		raise Exception("Unknown kernel_type: accepted types [presence, spectrum, intersection]")

	num_cores = mp.cpu_count()
	sqrt_num_cores = int(math.sqrt(num_cores))
	number_of_p_grams = maxNgramLen - minNgramLen + 1

	# create dictionaries (parallel)
	print("NumCores: {} {}".format(mp.cpu_count(), num_cores))
	pool = mp.Pool(num_cores)
	result = pool.map(createDictionaries_helper, [[corpus_splitted, minNgramLen, maxNgramLen] for corpus_splitted in np.array_split(corpus, num_cores)])
	pool.close()

	dictionaries_list = []
	for intermediate_results in result:
		for dict_list in intermediate_results:
			dictionaries_list.append(dict_list)

	# computeKernel (parallel)
	row_split = np.array_split(dictionaries_list, sqrt_num_cores)
	column_split = np.array_split(dictionaries_list, sqrt_num_cores)

	global TRACE_ROWS, TRACE_COLUMNS
	TRACE_ROWS = len(row_split[0])
	TRACE_COLUMNS = len(column_split[0])

	index_i, index_j = 0, 0
	args_list = []
	for i in range(sqrt_num_cores):
		index_j = 0
		for j in range(sqrt_num_cores):
			row_indexes = (index_i, index_i + len(row_split[i]))
			column_indexes = (index_j, index_j + len(column_split[j]))
			args_list.append([row_indexes, column_indexes, row_split[i], column_split[j], number_of_p_grams, kernel_type])
			index_j += len(column_split[j])
		index_i += len(row_split[i])

	print("NumCores: {} {}".format(mp.cpu_count(), num_cores))
	pool = mp.Pool(num_cores)
	result = pool.map(computeKernel_helper, args_list)
	pool.close()

	# combine the results
	kernelMatrix = np.zeros((number_of_p_grams, len(corpus), len(corpus)))
	cnt = len(result)
	for row_indexes, column_indexes, result_matrix in result:
		start = time.time()
		kernelMatrix[:, row_indexes[0]: row_indexes[1], column_indexes[0]: column_indexes[1]] = result_matrix
		end = time.time()
		cnt -= 1
		print("Matrix transfer took {} s. Remaining {}".format(end - start, cnt))

	for i in range(number_of_p_grams):
		kernelMatrix[i] = kernelMatrix[i] + np.transpose(kernelMatrix[i]) - np.diag(np.diag(kernelMatrix[i]))

	return kernelMatrix.astype(int)

def cleanCorpus(corpus):
	corpus = [sample.strip() for sample in corpus]
	corpus = [sample.lower() for sample in corpus]
	corpus = [re.sub(r'\s+', ' ', sample) for sample in corpus]
	return corpus

def computeAndSave(minNgramLen, maxNgramLen, kernel_type, path_to_directory, file_list, path_to_save):
	final_corpus = []

	for file_name in file_list:
		with open(os.path.join(path_to_directory, file_name)) as f:
			corpus = f.read().splitlines()
			corpus = cleanCorpus(corpus=corpus)
		for sample in corpus:
			final_corpus.append(sample)

	final_corpus = final_corpus

	start = time.time()
	res = computeKernelMatrix(corpus=final_corpus, minNgramLen=minNgramLen, maxNgramLen=maxNgramLen, kernel_type=kernel_type)
	end = time.time()

	print(res)
	print("time for {} samples and {}-{} ngrams => {} s".format(len(final_corpus), minNgramLen, maxNgramLen, end - start))
	print(np.sum(res, axis=0))

	np.save(path_to_save, res)
	return res

def test():
	with open("example.txt") as f:
		corpus = f.read().splitlines()
		corpus = cleanCorpus(corpus=corpus)
	res = computeKernelMatrix(corpus=corpus, minNgramLen=3, maxNgramLen=5, kernel_type="presence")
	print("per n-gram matrix:")
	print(res)
	final = np.sum(res, axis=0)
	print("final:")
	print(final)

if __name__ == '__main__':
	#test()
	#exit(-1)
	minNgramLen = 3
	maxNgramLen = 8
	kernel_type = "presence"
	res = computeAndSave(minNgramLen=minNgramLen,
						 maxNgramLen=maxNgramLen,
						 kernel_type=kernel_type,
						 path_to_directory="",
						 file_list=["all_sentences_without_cleaning.txt"],
						 path_to_save="all_sentences_without_cleaning_{}_{}_{}.npy".format(kernel_type, minNgramLen, maxNgramLen))

	print(res.shape)


