# SimpleHistoryExample.py
from __future__ import print_function
from __future__ import absolute_import

import requests
import pandas as pd
import numpy as np
from optparse import OptionParser
from sklearn.externals import joblib

def calcRSI(prices, n):
	totalGain = 0
	totalLoss = 0
	avgLoss = 0
	avgGain = 0
	rsi = []
	for day in range(30, len(prices)-1):
		if(day==30):
			for j in range(day-14, day):
				if(prices[j] > prices[j-1]):
					totalGain = totalGain + prices[j] - prices[j-1]
				else:
					totalLoss = totalLoss + prices[j-1] - prices[j]
			avgGain = totalGain/14
			avgLoss = totalLoss/14
			if(avgLoss == 0):
				rsiValue = 100
			else:
				rs = avgGain/avgLoss
				rsiValue = 100 - 100/(1+rs)
			rsi.append(rsiValue)
		else:
			gain = 0
			loss = 0
			if(prices[day] > prices[day-1]):
				gain = prices[day] - prices[day-1]
			else:
				loss = prices[day-1] - prices[day]
			avgGain = (avgGain*13 + gain)/14
			avgLoss = (avgLoss*13 + loss)/14
			if(avgLoss == 0):
				rsiValue = 100
			else:
				rs = avgGain/avgLoss
				rsiValue = 100-100/(1+rs)
			rsi.append(rsiValue)
	rsi = rsi[20:(n-60)]
	rsi = np.asarray(rsi)
	return rsi

def calcStochOscWilliams(prices, n):
	stochOsc = []
	williams = []
	for i in range(30, len(prices)-1):
		lowest = 1000000000
		highest = 0
		day = i
		for j in range(day-14, day):
			if(prices[j] > highest):
				highest = prices[j]
			if(prices[j] < lowest):
				lowest = prices[j]
		currentPrice = prices[day]
		stochOscValue = 100*(currentPrice-lowest)/(highest-lowest)
		williamsValue = (highest-currentPrice)/(highest-lowest) * -100
		stochOsc.append(stochOscValue)
		williams.append(williamsValue)
	stochOsc = stochOsc[20:(n-60)]
	williams = williams[20:(n-60)]
	stochOsc = np.asarray(stochOsc)
	williams = np.asarray(williams)
	return stochOsc, williams

def calcMACDSignalLine(prices, n):
	emaTwentySix = []
	emaTwelve = []
	signalLine = []
	multiplier = 2/(26+1)
	for day in range(30, len(prices)-1):
		if(day == 30):
			emaTwentySix.append(np.mean(prices[4:30]))
		else:
			emaTwentySix.append(multiplier*(prices[day]-emaTwentySix[day-31])+emaTwentySix[day-31])
	multiplier = 2/(12+1)
	for i in range(30, len(prices)-1):
		day = i
		if(day == 30):
			emaTwelve.append(np.mean(prices[18:30]))
		else:
			emaTwelve.append(multiplier*(prices[day]-emaTwelve[day-31])+emaTwelve[day-31])
	emaTwentySix = np.asarray(emaTwentySix)
	emaTwelve = np.asarray(emaTwelve)    
	macd = emaTwelve-emaTwentySix
	multiplier = 2/(9+1)
	for i in range(9, len(macd)-1):
		day = i
		if(day == 9):
			signalLine.append(np.mean(macd[0:9]))
		else:
			signalLine.append(multiplier*(macd[day]-signalLine[day-10])+signalLine[day-10])
	macd = macd[20:(n-60)]
	signalLine = signalLine[10:(n-70)]   
	macd = np.asarray(macd)
	signalLine = np.asarray(signalLine) 
	return macd, signalLine

def calcPROC(prices, n):
	procThirty = []
	procSeven = []
	for i in range(30, len(prices)-1):
		procThirty.append((prices[i]-prices[i-25])/prices[i-25])
		procSeven.append((prices[i]-prices[i-7])/prices[i-7]) 
	procSeven = procSeven[20:(n-60)]
	procThirty = procThirty[20:(n-60)]  
	procThirty = np.asarray(procThirty)
	procSeven = np.asarray(procSeven)
	return procSeven, procThirty
	
def getGICS(gicsNum):
	gics = "XLC"
	if(gicsNum == 10):
		gics = "XLE"
	if(gicsNum == 15):
		gics = "XLB"
	if(gicsNum== 20):
		gics = "XLI"
	if(gicsNum == 25):
		gics = "XLY"
	if(gicsNum == 30):
		gics = "XLP"
	if(gicsNum == 35):
		gics = "XLV"
	if(gicsNum == 40):
		gics = "XLF"
	if(gicsNum == 45):
		gics = "XLK"
	if(gicsNum == 50):
		gics = "XLC"
	if(gicsNum == 55):
		gics = "XLU"
	if(gicsNum == 60):
		gics = "XLRE"
	return gics

def calcOBV(prices, volumes, n):
	obvSeven = []
	obvThirty = []
	obv = []
	for i in range(0, len(prices)-1):
		if(i == 0):
			obv.append(0)
		else:
			pctChange = (prices[i]-prices[i-1])/prices[i-1]
			obv.append(volumes[i]*pctChange + obv[i-1])
	for i in range(30, len(prices)-1):
		obvSeven.append((obv[i]-obv[i-7])/obv[i-7])
		obvThirty.append((obv[i]-obv[i-25])/obv[i-25])
	obvSeven = obvSeven[20:(n-60)]
	obvThirty = obvThirty[20:(n-60)]
	obvSeven = np.asarray(obvSeven)
	obvThirty = np.asarray(obvThirty)
	return obvSeven, obvThirty

def getHistoricalData(ticker):
	r = requests.get('https://api.iextrading.com/1.0/stock/' + ticker + '/chart/5y')
	r = r.json()
	volumes = []
	prices = []
	highs = []
	lows = []
	for date in r:
		volumes.append(date.get('volume'))
		highs.append(date.get('high'))
		lows.append(date.get('low'))
		prices.append(date.get('close'))
	return prices, volumes, highs, lows

def getLabels(prices, highs,lows, change, days, n):
	compLabels = []
	if(n > 100):
		i = 0
		for day in prices:
			if i > 49:
				if i < n-30:
					k = 0
					if(change > .03):
						for j in range(1, days):
							if(k == 50):
								k = 50
							elif(highs[i+j] == None or lows[i+j] == None):
								print("NONE")
							elif((highs[i+j]-prices[i])/prices[i] > change):
								compLabels.append(-1)
								k = 50
							elif((lows[i+j]-prices[i])/prices[i] < -change):
								compLabels.append(1)
								k = 50
							elif(j == days-1):
								compLabels.append(0)
								k = 50
					else:
						if((prices[i+days]-prices[i])/prices[i] > change):
							compLabels.append(-1)
						elif((prices[i+days]-prices[i])/prices[i] < -change):
							compLabels.append(1)
						else:
							compLabels.append(0)
								
			i = i+1
	return compLabels


def main():
	allFeatures = []
	allFeatures = np.asarray(allFeatures)
	allLabels = []
	first = 0
	
	change = .01
	days = 20
	filename = 'new_finalized_model_' + str(days) + 'day_' + str(change) + '_pct.sav'
	
	allLabels = np.asarray(allLabels)
	import csv
	import pickle
	with open('ticker_sectors.data', 'rb') as f:
    		tickerSectors = pickle.load(f)
	companies = tickerSectors[0]
	sectors = tickerSectors[1]
	for i in range(0, len(companies)):
		comp = companies[i]
		print(comp)
		prices, volumes, highs, lows = getHistoricalData(comp)

		sec = sectors[i]
		secPrices, secVolumes, secHighs, secLows = getHistoricalData(sec)

		n = len(prices)
		compLabels = getLabels(prices, highs, lows, change, days, n)

		if(n == len(secPrices)):
			compLabels = pd.DataFrame(data = compLabels, columns = ['Next Month Change'])
			compLabels = np.array(compLabels['Next Month Change'])
			trainLabelsComp = compLabels[0:800]
			testLabelsComp = compLabels[840:]
			rsi = calcRSI(prices, n)
			macd, signalLine = calcMACDSignalLine(prices, n)
			stochOsc, williams = calcStochOscWilliams(prices, n)
			procSeven, procThirty = calcPROC(prices, n)
			obvSeven, obvThirty = calcOBV(prices, volumes, n)
			secRsi = calcRSI(secPrices, n)
			secProcSeven, secProcThirty = calcPROC(secPrices, n)

			if(macd.shape==secRsi.shape):
				features = np.concatenate(([rsi], [macd], [signalLine], [stochOsc], [williams], [procSeven], [procThirty], [obvSeven], [obvThirty], [secRsi], [secProcSeven], [secProcThirty]), axis=0)
				features = np.transpose(features)
				trainFeaturesComp = features[0:800]
				testFeaturesComp = features[840:]
				
				if(first == 0):
					train_features = np.asarray(trainFeaturesComp)
					test_features = np.asarray(testFeaturesComp)
					train_labels = np.asarray(trainLabelsComp)
					test_labels = np.asarray(testLabelsComp)
					first = 1
				else:
					if(testLabelsComp.shape[0] == testFeaturesComp.shape[0]):
						train_features = np.concatenate((train_features,trainFeaturesComp))
						test_features = np.concatenate((test_features,testFeaturesComp))
						train_labels = np.concatenate((train_labels,trainLabelsComp))
						test_labels = np.concatenate((test_labels,testLabelsComp))
					print(test_labels.shape)
					print(test_features.shape)
		else:
			print("MISMATCH")


	print('Training Features Shape:', train_features.shape)
	print('Training Labels Shape:', train_labels.shape) 
	print('Testing Features Shape:', test_features.shape)
	print('Training Labels Shape:', test_labels.shape)          
	baseline_preds = 1
	baseline_errors = abs(baseline_preds - test_labels)
	print('Average baseline error: ', round(np.mean(baseline_errors), 8))
				
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.metrics import confusion_matrix
	rf = RandomForestRegressor(n_estimators=65, random_state = 42)
	rf.fit(train_features, train_labels);
	joblib.dump(rf, filename)
		
	# loaded_model = joblib.load(filename)
	# result = loaded_model.score(X_test, Y_test)
	# print(result)
		
	predictions = rf.predict(test_features)
	print(predictions)
	for i in range(len(predictions)):
		if(predictions[i] > .33):
			predictions[i] = 1
		elif(predictions[i] > -.33):
			predictions[i] = 0
		else:
			predictions[i] = -1
	target_names = ['-1', '0', '1']
	from sklearn.metrics import classification_report
	print(classification_report(test_labels, predictions, target_names=target_names))
	confusionMatrix = confusion_matrix(test_labels, predictions)
	print('Confusion Matrix: ', confusionMatrix)
	
			

if __name__ == "__main__":
	print("SimpleHistoryExample")
	try:
		main()
	except KeyboardInterrupt:
		print("Ctrl+C pressed. Stopping...")


