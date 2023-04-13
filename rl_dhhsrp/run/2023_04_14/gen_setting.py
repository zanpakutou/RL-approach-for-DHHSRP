for obj in ['visit']:
	for nb_scen in ['75']:
		for inter_arrival_rate in ['40', '100']:
			for _type in ['U']:
				for nb_nurse in ['24']:
					for instance_id in range(950, 980):
						for switch in ['0', '1']:
							command = "--obj " + obj + " --nb_scenario " + nb_scen +  ' --arr_rate ' + inter_arrival_rate \
							+ ' --instance_type ' + _type + " --nb_nurse " + nb_nurse + " --output_folder sba/" \
							+ " --instance_id " + str(instance_id) + " --switch_CH " + switch
							print(command)