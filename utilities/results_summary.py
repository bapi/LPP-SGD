from operator import itemgetter


def get_data(result_json, tag, wrt):
    data = []
    for r in result_json:
        if r['tag'] == tag:
            data.append((r[wrt], r['val']))
    return data


def results_summary(result, args):
    summary_dict = []
    tags = ["TrainLoss", "TrainAcc_1", "TestLoss", "TestAcc_1"]
    for wrt in ['ep', 'time']:
        for tag in tags:
            if 'Acc_' in tag:
                tag = tag.replace('Acc_', 'Acc@')
            data = get_data(result, tag, wrt)
            data.sort(key=itemgetter(0))
            if wrt == 'ep':
                if "Loss" in tag:
                    summary_dict.append({tag: min(data, key=itemgetter(1))[1]})
                elif "Acc" in tag:
                    summary_dict.append({tag: max(data, key=itemgetter(1))[1]})
            if wrt == 'time':
                summary_dict.append(
                    {"TotalTime": max(data, key=itemgetter(0))[0]})
    print("Rank: ", args.commrank, ", Results Summary: ", summary_dict)
