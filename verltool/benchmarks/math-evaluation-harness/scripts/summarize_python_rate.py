import json
import glob
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--data_names", type=str, default="gsm8k,minerva_math,svamp,asdiv,mawps")
    parser.add_argument("--prompt_type", type=str, default="tool_math_qwen")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    summarize_results(args.result_dir, args.data_names, args.prompt_type, args.temperature, args.split)


def summarize_results(result_dir, data_names, prompt_type, temperature, split):
    data_list = data_names.split(',')

    # read the result
    results = []
    for data_name in data_list:
        files = []
        for f in glob.glob(f"{result_dir}/{data_name}/{split}_{prompt_type}*.json"):
            if "0.0" in f and "metrics" not in f: # TODO: filter
                files.append(f)
        
        assert len(files) == 1, f"Found {len(files)} files for {data_name}"
        with open(files[0], 'r') as f: # TODO: read file dict and compute python rate 
            metrics = json.load(f)
            results.append(metrics)
    
    data_list.append("avg")
    results.append({
        "acc": sum([result["acc"] for result in results]) / len(results),
    })
    
    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))
    print(" & ".join([f"{result['acc']:.1f}" for result in results]))


if __name__ == "__main__":
    main()
