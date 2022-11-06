for obj in ['patient','visit']:
	for inter_arrival_rate in ['90', '150', '240', '360']:
		for _type in ['uniform', 'cluster']:
			for default_config in ['0']:
				for lr in ['0.00001']:
					for nb_nurse in ['1', '6', '12']:
						for discount_factor in ['0.997']:
							for cap_heur in ['True', 'False']:
								timestep='15000000'
								batch_size='512'

								command = timestep + ' ' + batch_size\
								+ ' ' + discount_factor + ' ' + _type + ' ' + inter_arrival_rate\
								+ ' ' + obj + ' ' + lr + ' ' + cap_heur\
								+ ' ' + nb_nurse
									
								print(command)
'''
for obj in ['patient','visit']:
    for nb_scen in ['5', '10', '20', '30']:
        for inter_arrival_rate in ['150', '240', '360']:
            for _type in ['uniform', 'cluster', 'simplify']:
                command = "--obj " + obj + " --nb_scenario " + nb_scen +  ' --inter_arrival_rate ' + inter_arrival_rate \
                    + ' --instance_type ' + _type + " --output_folder sba/" + _type
                print(command)
'''