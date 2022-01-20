import pandas as pd


def append_float(target, v):
	m = str(float(v))[:6]
	target += '\t&\t%s' % m
	return target

def pprint_error(df_path, time_path=None, exp_names=None, compare_targets=None, metrics=None):
	df = pd.read_csv(df_path)
	exp_names_dict = {
		"ours": "IDNeRF (Ours)",
		"monte_carlo_nerf_surface": "MC",
		"monte_carlo_env_map": "MC + Env.map",
	}
	if exp_names is None:
		exp_names = ["monte_carlo_nerf_surface", "monte_carlo_env_map", "ours"]
	if compare_targets is None:
		compare_targets = ["diffuse", "specular", "image"]
	if metrics is None:
		metrics = ["mse", "psnr", "ssim"]

	time_df = None
	if time_path is not None:
		time_df = pd.read_csv(time_path)

	table_str = ""


	for exp_name in exp_names:
		table_str += exp_names_dict[exp_name]
		for compare_target in compare_targets:
			data = df.loc[(df['exp_name'] == exp_name) & (df['compare_target'] == compare_target)]

			for metric in metrics:
				table_str = append_float(table_str, data[metric])

		if time_df is not None:
			elapsed_time = time_df.loc[(time_df['exp_name']==exp_name)]["time_per_step"]
			table_str = append_float(table_str, elapsed_time)

		table_str += "\\\\\n"
	print(table_str)

# pprint_error("../logs_eval/final_config_lindisp_equal_sample/error.csv",
# 			 time_path="../logs/final_config_lindisp_equal_sample/time.csv")
pprint_error("../../logs_eval/final_config_lindisp_equal_sample/error.csv")