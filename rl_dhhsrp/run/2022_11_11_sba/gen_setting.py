for obj in ['patient','visit']:
	for nb_scen in ['75']:
		for inter_arrival_rate in ['90', '150', '240', '360']:
			for _type in ['uniform', 'cluster']:
				for nb_nurse in ['1', '6', '12']:
					command = "--obj " + obj + " --nb_scenario " + nb_scen +  ' --arr_rate ' + inter_arrival_rate \
					+ ' --instance_type ' + _type + " --nb_nurse " + nb_nurse + " --output_folder sba/" + _type 
					print(command)