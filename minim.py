with open("test_dil_1015_pos.dat2",'r') as file:
	mini=[]
	for i in range(1015):
		print(i)
		store=[]
		for l in file:
			first=l.split()
			if int(first[0])==i:
				store.append(float(first[2]))
			else:
				break
		mini.append(min(store))
#		print(mini)
print(mini)
print(len(mini))
sum=sum(float(sub) for sub in mini)
print(sum/len(mini))	
