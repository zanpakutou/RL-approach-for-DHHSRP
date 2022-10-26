for obj in ['patient','visit']:
    for nb_scen in ['5', '10', '20', '30']:
        for inter_arrival_rate in ['150', '240', '360']:
            for _type in ['uniform', 'cluster', 'simplify']:
                command = "--obj " + obj + " --nb_scenario " + nb_scen +  ' --inter_arrival_rate ' + inter_arrival_rate \
                    + ' --instance_type ' + _type + " --output_folder " + _type
                print(command)
                