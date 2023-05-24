for obj in ['patient']:
	for nb_scen in ['75']:
		for inter_arrival_rate in ['255', '100']:
			for _type in ['U']:
				for nb_nurse in ['12']:
					for instance_id in range(950, 980):
						command = "--obj " + obj + " --nb_scenario " + nb_scen +  ' --arr_rate ' + inter_arrival_rate \
						+ ' --instance_type ' + _type + " --nb_nurse " + nb_nurse + " --output_folder sba/" \
						+ " --instance_id " + str(instance_id)
						print(command)