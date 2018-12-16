from __future__ import print_function
from __future__ import absolute_import

import requests
import pandas as pd
import numpy as np
from optparse import OptionParser
from sklearn.externals import joblib
import csv

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
		gics = "xLRE"
	return gics
def calcRSI(prices):
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
	return np.asarray(rsi[len(rsi)-1])

def calcStochOscWilliams(prices):
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
	return np.asarray(stochOsc[len(stochOsc)-1]), np.asarray(williams[len(williams)-1])

def calcMACDSignalLine(prices):
	emaTwentySix = []
	emaTwelve = []
	signalLine = []
	multiplier = 2.0/(26+1)
	for day in range(30, len(prices)-1):
		if(day == 30):
			emaTwentySix.append(np.mean(prices[4:30]))
		else:
			emaTwentySix.append(multiplier*(prices[day]-emaTwentySix[day-31])+emaTwentySix[day-31])
	multiplier = 2.0/(12+1)
	for i in range(30, len(prices)-1):
		day = i
		if(day == 30):
			emaTwelve.append(np.mean(prices[18:30]))
		else:
			emaTwelve.append(multiplier*(prices[day]-emaTwelve[day-31])+emaTwelve[day-31])
	emaTwentySix = np.asarray(emaTwentySix)
	emaTwelve = np.asarray(emaTwelve)    
	macd = emaTwelve-emaTwentySix
	multiplier = 2.0/(9+1)
	for i in range(9, len(macd)-1):
		day = i
		if(day == 9):
			signalLine.append(np.mean(macd[0:9]))
		else:
			signalLine.append(multiplier*(macd[day]-signalLine[day-10])+signalLine[day-10])
	return np.asarray(macd[len(macd)-1]), np.asarray(signalLine[len(signalLine)-1])

def calcPROC(prices):
	procThirty = []
	procSeven = []
	for i in range(30, len(prices)-1):
		procThirty.append((prices[i]-prices[i-30])/prices[i-30])
		procSeven.append((prices[i]-prices[i-7])/prices[i-7]) 
	return np.asarray(procSeven[len(procSeven)-1]), np.asarray(procThirty[len(procThirty)-1])

def calcOBV(prices, volumes):
	obvSeven = []
	obvThirty = []
	obv = []
	for i in range(0, len(prices)-1):
		if(i == 0):
			obv.append(0)
		else:
			if(prices[i] > prices[i-1]):
				obv.append(obv[i-1] + volumes[i])
			else:
				obv.append(obv[i-1] - volumes[i])
	for i in range(30, len(prices)-1):
		obvSeven.append(obv[i]-obv[i-7])
		obvThirty.append(obv[i]-obv[i-30])
	return obvSeven[len(obvSeven)-1], obvThirty[len(obvThirty)-1]

def makePredictions(models):
	cs = []
	allMetrics = []
	import pickle
	with open('ticker_sectors.data', 'rb') as f:
		tickerSectors = pickle.load(f)
	loaded_models = []
	for model in models:
		currentMod = joblib.load('new_finalized_model_' + str(model[0]) + 'day_' + str(model[1]) + '_pct.sav')
		loaded_models.append(currentMod)
	companies = tickerSectors[0]
	sectors = tickerSectors[1]

	for i in range(0, len(companies)):
		comp = companies[i]
		print(comp)
		sec = sectors[i]
		r = requests.get('https://api.iextrading.com/1.0/stock/' + comp + '/chart/1y')
		r = r.json()
		volumes = []
		prices = []
		for date in r:
			volumes.append(date.get('volume'))
			prices.append(date.get('close'))
		macd, signalLine = calcMACDSignalLine(prices)
		stochOsc, williams = calcStochOscWilliams(prices)
		obvSeven, obvThirty = calcOBV(prices, volumes)
		procSeven, procThirty = calcPROC(prices)
		rsi = calcRSI(prices)
		
		sec = requests.get('https://api.iextrading.com/1.0/stock/' + sec + '/chart/1y')
		sec = sec.json()
		secPrices = []
		for date in sec:
			secPrices.append(date.get('close'))
		secRSI = calcRSI(secPrices)
		secProcSeven, secProcThirty = calcPROC(secPrices)
					
		features = np.concatenate(([rsi], [macd], [signalLine], [stochOsc], [williams], [procSeven], [procThirty], [obvSeven], [obvThirty], [secRSI], [secProcSeven], [secProcThirty]), axis=0)
		allMetrics.append([])
		allMetrics[i].append(comp)
		allMetrics[i].append(sectors[i])
		allMetrics[i].append(rsi)
		allMetrics[i].append(macd)
		allMetrics[i].append(signalLine)
		allMetrics[i].append(stochOsc)
		allMetrics[i].append(williams)
		allMetrics[i].append(procSeven)
		allMetrics[i].append(procThirty)
		allMetrics[i].append(secRSI)
		allMetrics[i].append(secProcSeven)
		allMetrics[i].append(secProcThirty)

		k = 0
		for model in models:
			loaded_model = loaded_models[k]
			cs = loaded_model.predict([features])[0]
			allMetrics[i].append(cs)
			k = k+1
	return allMetrics, companies



def main():
	allTops = []
	models = [[5, .01], [5, .04], [20, .01], [20, .04]]
	allMetrics, companies = makePredictions(models)
	with open("allMetrics.csv", "w") as f:
    		writer = csv.writer(f)
    		writer.writerows(allMetrics)
	allMetrics = np.asarray(allMetrics)
	companyList = []
	csList = []
	days = []
	percents = []



	companyList = np.asarray(companyList)
	csList = np.asarray(csList)
	days = np.asarray(days)
	percents = np.asarray(percents)
	companyList = np.concatenate(([companyList], [csList], [days]), axis=0)
	companyList = np.concatenate((companyList, [percents]), axis=0)


			

		
if __name__ == "__main__":
	print("SimpleHistoryExample")
	try:
		main()
	except KeyboardInterrupt:
		print("Ctrl+C pressed. Stopping...")





