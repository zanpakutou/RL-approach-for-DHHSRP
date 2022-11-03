'''for obj in ['patient','visit']:
	for inter_arrival_rate in ['60']:
		for _type in ['uniform']:
			for default_config in ['0']:
				for lr in ['0.000001']:
					for trans in ['partial', 'full']:
						for discount_factor in ['0.997']:
							episodes='50000'
							batch_size='512'
							NN_size='256'

							command = episodes + ' ' + batch_size + ' ' + NN_size\
							+ ' ' + discount_factor + ' ' + _type + ' ' + inter_arrival_rate\
							+ ' ' + obj + ' ' + lr + ' ' + trans
								
							print(command)'''
for obj in ['patient','visit']:
    for nb_scen in ['5', '10', '20', '30']:
        for inter_arrival_rate in ['150', '240', '360']:
            for _type in ['uniform', 'cluster', 'simplify']:
                command = "--obj " + obj + " --nb_scenario " + nb_scen +  ' --inter_arrival_rate ' + inter_arrival_rate \
                    + ' --instance_type ' + _type + " --output_folder sba/" + _type
                print(command)
                
