import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# define constants
conf_groups = {
    "Low": range(1, 5), #2493
    "Medium": range(5, 7), #7127 
    "High": range(7, 8) #9281
}

choice_types = {
    "Unsafe": 0.5,
    "Safe": 0.9 
}

option_chosen = {
    "Risky": 1,
    "Certain": 0
}

# load data
file_path = r"./phase2_data_updated.xlsx"
df = pd.read_excel(file_path)

# get relevant data
deltaSV = df['deltaSV'].values
SPTdeltaSV = df['SPTdeltaSV'].values
choice = df['Gamble'].values
confidence = df['Confidence'].values
reaction_time = df['rt'].values

# define bins and bin centers for deltaSV values
bins = np.linspace(-1.0, 1.0, 7)
bin_centers = []
for i in range(len(bins) - 1):      
    bin_center = (bins[i] + bins[i + 1]) / 2
    bin_centers.append(bin_center)
# add bins to data set
df["deltaSV_bin"] = pd.cut(df["deltaSV"], bins=bins, labels=bin_centers, include_lowest=True).astype(float)
df["SPTdeltaSV_bin"] = pd.cut(df["SPTdeltaSV"], bins=bins, labels=bin_centers, include_lowest=True).astype(float)

# define bins and bin centers for reaction time values
bins_rt = np.linspace(600, 2200, 8)
bin_centers_rt = []
for i in range(len(bins_rt) - 1):      
    bin_center_rt = (bins_rt[i] + bins_rt[i + 1]) / 2
    bin_centers_rt.append(bin_center_rt)
# add bins to data set
df["rt_bin"] = pd.cut(df["rt"], bins=bins_rt, labels=bin_centers_rt, include_lowest=True).astype(float)
# get reaction time groups
rt_groups = {
    "Low": [bin_centers_rt[0], bin_centers_rt[1], bin_centers_rt[2]], #2102
    "Medium": [bin_centers_rt[3], bin_centers_rt[4]], #6900
    "High": [bin_centers_rt[5], bin_centers_rt[6]] #5706
}

# option chosen masks
risky_chosen = df["Gamble"] == option_chosen["Risky"]
certain_chosen = df["Gamble"] == option_chosen["Certain"]

# choice type masks
unsafe_choice_mask = df["P_Gamble"] == choice_types["Unsafe"]
safe_choice_mask = df["P_Gamble"] == choice_types["Safe"]

# filter data frame for risky and safe choices
unsafe_df = df[unsafe_choice_mask].copy()
safe_df = df[safe_choice_mask].copy()



# calculate probability of choosing the risky 
def calculate_prob_choosing_risky_all(deltaSV, choice, bins, low_mask, med_mask, high_mask):
    ''' ACCURACY: prob of choosing risky choice (dependent on delta SV)
        if SV is positive, should choose risky choice
        if SV is negative, should choose safe choice '''
    prob_risky_all = []
    prob_risky_low_c = []
    prob_risky_med_c = []
    prob_risky_high_c = []
    sem_all = []
    sem_low_c = []
    sem_med_c = []
    sem_high_c = []

    # calculate SEM 
    def calculate_sem(data):
        return np.std(data, ddof=1) / np.sqrt(len(data))

    def get_prob_and_sem(bin_mask, choice):
        # use Gamble column: what was the average choice for each of the bins?
        # bins are dependent on deltaSV
        if np.any(bin_mask):
            prob_risky = np.mean(choice[bin_mask] == 1) # risky = 1
            sem = calculate_sem(choice[bin_mask] == 1)
        else:
            prob_risky = 0
            sem = 0
            
        return prob_risky, sem
    
    for i in range(len(bins) - 1):
        bin_mask = ((deltaSV >= bins[i]) & (deltaSV < bins[i + 1]))       
        
        # compute probabilities for current bin
        # for all
        prob_any, sem_any = get_prob_and_sem(bin_mask, choice)
        prob_risky_all.append(prob_any)
        sem_all.append(sem_any)
            
        # for low conf
        low_c_bin_mask = bin_mask & low_mask
        
        prob_low, sem_low = get_prob_and_sem(low_c_bin_mask, choice)
        prob_risky_low_c.append(prob_low)
        sem_low_c.append(sem_low)
        
        # for med conf
        med_c_bin_mask = bin_mask & med_mask
        
        prob_med, sem_med = get_prob_and_sem(med_c_bin_mask, choice)
        prob_risky_med_c.append(prob_med)
        sem_med_c.append(sem_med)
        
        # for high conf
        high_c_bin_mask = bin_mask & high_mask
        
        prob_high, sem_high = get_prob_and_sem(high_c_bin_mask, choice)
        prob_risky_high_c.append(prob_high)
        sem_high_c.append(sem_high)
    
    return (prob_risky_all, sem_all,
            prob_risky_low_c, sem_low_c,
            prob_risky_med_c, sem_med_c,
            prob_risky_high_c, sem_high_c)

# bootstrap errors
def bootstrap_errors(ydata_list, n_bootstrap=1000, ci=68):
    error_list = []

    for ydata in ydata_list:
        ydata = np.array(ydata)
        n_points = len(ydata)
        boot_means = np.zeros((n_bootstrap, n_points))

        # bootstrap
        for i in range(n_bootstrap):
            resample_idx = np.random.choice(np.arange(n_points), size=n_points, replace=True)
            resample = ydata[resample_idx]
            boot_means[i] = resample

        # compute standard error or CI
        lower = (100 - ci) / 2
        upper = 100 - lower
        error = np.percentile(boot_means, [upper], axis=0) - np.percentile(boot_means, [lower], axis=0)
        error = error.squeeze() / 2  # divide by 2 to get error bars above/below
        error_list.append(error)

    return error_list

# standard function to plot data
def plot_data(xdata, ydata_list, error_list=None, labels=None, title=None, \
               xlabel=None, ylabel=None, colors=None, filename=None):
    fig, ax = plt.subplots(figsize=(4, 4)) # create figure plot

    # bootstrap errors if no error list provided
    if error_list is None:
        error_list = bootstrap_errors(ydata_list)
    
    # plot each condition
    for i, (ydata, label, color) in enumerate(zip(ydata_list, labels, colors)): 
        yerr = error_list[i]
        ax.errorbar(xdata, ydata, yerr=yerr, label=label, color=color, marker='o', 
                    linestyle='-', linewidth=2, markersize=4, capsize=2)

    #ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.axhline(0.5, color='grey', linestyle='--')
    
    #if labels and len(labels) > 1:
    #    ax.legend(fontsize=8, frameon=False)

    plt.tight_layout()

    # show or save plot
    if filename:
        plt.savefig(filename, bbox_inches="tight")
        print(f"plot saved: {filename}")
    else:
        plt.show()
    
    plt.close(fig)



# ---------- Probabilities of Choice Risky Option Based on Choice Type and DeltaSV Type ------------ # 

def deltaSV_vs_prob(local_df, deltaSV, choice, deltaSV_title, choice_title, deltaSV_filename, choice_filename, conf_measure):
    # get confidence masks OR reaction time masks
    if (conf_measure == "confidence"):
        low_conf = local_df["Confidence"].isin(conf_groups["Low"])
        med_conf = local_df["Confidence"].isin(conf_groups["Medium"])
        high_conf = local_df["Confidence"].isin(conf_groups["High"])
    if (conf_measure == "RT"):
        low_conf = local_df["rt_bin"].isin(rt_groups["Low"])
        med_conf = local_df["rt_bin"].isin(rt_groups["Medium"])
        high_conf = local_df["rt_bin"].isin(rt_groups["High"])

    # compute probs
    (prob_risky, sem,
        prob_risky_low_c, sem_low_c,
        prob_risky_med_c, sem_med_c,
        prob_risky_high_c, sem_high_c) = calculate_prob_choosing_risky_all(
                    deltaSV, choice, bins, low_conf, med_conf, high_conf)
    
    # add binned probabilities to data set for all choice types
    if (choice_filename == "all"): 
        bin_prob_dict = dict(zip(bin_centers, prob_risky)) # {bin_center: probability}
        
        if (deltaSV_filename == "deltaSV"): 
            local_df["prob_choosing_risky_bin"] = local_df["deltaSV_bin"].map(bin_prob_dict)
        if (deltaSV_filename == "spt_deltaSV"):
            local_df["prob_choosing_risky_bin_SPT"] = local_df["SPTdeltaSV_bin"].map(bin_prob_dict)
       
        output_path = r"./phase2_data_updated.xlsx"
        local_df.to_excel(output_path, index=False)
    
    # plot probs
    plot_data(bin_centers, 
        [prob_risky, prob_risky_low_c, prob_risky_med_c, prob_risky_high_c], 
        error_list=[sem, sem_low_c, sem_med_c, sem_high_c],
        labels=[f"all {conf_measure}", f"low {conf_measure}", f"medium {conf_measure}", f"high {conf_measure}"], 
        colors=['black', 'red', 'blue', 'green'],
        title=f"Conditioned Psychometric Curve for P(risky)={choice_title}", 
        xlabel=f"{deltaSV_title}", 
        ylabel='Probability of Choosing Risky Option',
        filename=rf"./plots (final)\psychometric_analysis\prob_vs_{deltaSV_filename}_{choice_filename}_{conf_measure}2.png"
    )

def psychometric_analysis():
    # get plots for deltaSV
    deltaSV_vs_prob(df, deltaSV, choice, "Subjective Value Difference", "All", "deltaSV", "all", "confidence") # all choice types (unsafe and safe)
    deltaSV_vs_prob(unsafe_df, unsafe_df["deltaSV"].values, # unsafe choice type (P(risky) = 0.5)
                        unsafe_df["Gamble"].values, "Subjective Value Difference", "0.5", "deltaSV", "0.5", "confidence") 
    deltaSV_vs_prob(safe_df, safe_df["deltaSV"].values, # safe choice type (P(risky) = 0.9))
                        safe_df["Gamble"].values, "Subjective Value Difference", "0.9", "deltaSV", "0.9", "confidence")

    # get plots for SPTdeltaSV
    deltaSV_vs_prob(df, SPTdeltaSV, choice, "SPT Subjective Value Difference", "All", "spt_deltaSV", "all", "confidence") # all choice types (unsafe and safe)
    deltaSV_vs_prob(unsafe_df, unsafe_df["SPTdeltaSV"].values, # unsafe choice type (P(risky) = 0.5)
                        unsafe_df["Gamble"].values, "SPT Subjective Value Difference", "0.5", "spt_deltaSV", "0.5", "confidence") 
    deltaSV_vs_prob(safe_df, safe_df["SPTdeltaSV"].values, # safe choice type (P(risky) = 0.9))
                        safe_df["Gamble"].values, "SPT Subjective Value Difference", "0.9", "spt_deltaSV", "0.9", "confidence")

    # and also redo with RT instead of Conf
    deltaSV_vs_prob(df, deltaSV, choice, "Subjective Value Difference", "All", "deltaSV", "all", "RT") # all choice types (unsafe and safe)
    deltaSV_vs_prob(unsafe_df, unsafe_df["deltaSV"].values, # unsafe choice type (P(risky) = 0.5)
                        unsafe_df["Gamble"].values, "Subjective Value Difference", "0.5", "deltaSV", "0.5", "RT") 
    deltaSV_vs_prob(safe_df, safe_df["deltaSV"].values, # safe choice type (P(risky) = 0.9))
                        safe_df["Gamble"].values, "Subjective Value Difference", "0.9", "deltaSV", "0.9", "RT")




#------------------------------------EXTENDED ANALYSIS-----------------------------------------#

# add accuracies to data set
# accuracy dependent on deltaSV: positive deltaSV = risky = 1, negative deltaSV = safe = 0
df["correct"] = ((df["deltaSV"] > 0) & (risky_chosen)) | \
                ((df["deltaSV"] < 0) & (certain_chosen))
df["correct"] = df["correct"].astype(int)
output_path = r"./phase2_data_updated.xlsx"
df.to_excel(output_path, index=False)


# ANALYZE ACCURACY DEPENDENT ON CONFIDENCE
def confidence_vs_accuracy(local_df, conf, conf_title, choice_title, conf_filename, choice_filename):
    ''' EXPECTED RESULT: accuracy increases with confidence '''  
    accuracy_by_conf = local_df.groupby(conf)["correct"].mean()
    accuracy_by_conf_sem = local_df.groupby(conf)["correct"].sem()

    accuracy_by_conf_risky = local_df.loc[risky_chosen].groupby(conf)["correct"].mean()
    accuracy_by_conf_risky_sem = local_df.loc[risky_chosen].groupby(conf)["correct"].sem()

    accuracy_by_conf_certain = local_df.loc[certain_chosen].groupby(conf)["correct"].mean()
    accuracy_by_conf_certain_sem = local_df.loc[certain_chosen].groupby(conf)["correct"].sem()

    plot_data(accuracy_by_conf.index, 
              [accuracy_by_conf, accuracy_by_conf_risky, accuracy_by_conf_certain],
              error_list = [accuracy_by_conf_sem, accuracy_by_conf_risky_sem, accuracy_by_conf_certain_sem],
              labels=['all choice types', 'risky option chosen', 'certain option chosen'], 
              colors=['black', 'red', 'blue'],
              title=f"Calibration Curve ({conf_title}) for P(risky)={choice_title}",
              xlabel=f"{conf_title}", 
              ylabel='Accuracy',
              filename=rf"./plots (final)\conf_vs_acc_analysis\{conf_filename}_vs_accuracy_{choice_filename}2.png")

def confidence_accuracy_analysis():
    # get plots for confidence vs accuracy based on choice type
    confidence_vs_accuracy(df, "Confidence", "Confidence", "All", "confidence", "all")
    confidence_vs_accuracy(unsafe_df, "Confidence", "Confidence", "0.5", "confidence", "0.5")
    confidence_vs_accuracy(safe_df, "Confidence", "Confidence", "0.9", "confidence", "0.9")

    # get plots for reaction time vs accuracy based on choice type
    confidence_vs_accuracy(df, "rt_bin", "Response Time", "All", "rt", "all")
    confidence_vs_accuracy(unsafe_df, "rt_bin", "Response Time", "0.5", "rt", "0.5")
    confidence_vs_accuracy(safe_df, "rt_bin", "Response Time", "0.9", "rt", "0.9")



# ANALYZE DELTA SV DEPENDENT ON CONFIDENCE
def deltaSV_vs_confidence(local_df, conf, deltaSV_title, choice_title, conf_title, deltaSV_filename, choice_filename, conf_filename):
    ''' EXPECTED RESULT: ? '''
    
    if (deltaSV_filename == "deltaSV"):
        conf_by_sv = local_df.groupby("deltaSV_bin")[conf].mean()
        conf_by_sv_sem = local_df.groupby("deltaSV_bin")[conf].sem()
        deltaSV_groups = "deltaSV_bin"
    if (deltaSV_filename == "spt_deltaSV"):
        conf_by_sv = local_df.groupby("SPTdeltaSV_bin")[conf].mean()
        conf_by_sv_sem = local_df.groupby("SPTdeltaSV_bin")[conf].sem()
        deltaSV_groups = "SPTdeltaSV_bin"
    
    conf_by_sv_risky = local_df.loc[risky_chosen].groupby(deltaSV_groups)[conf].mean()
    conf_by_sv_risky_sem = local_df.loc[risky_chosen].groupby(deltaSV_groups)[conf].sem()

    conf_by_sv_certain = local_df.loc[certain_chosen].groupby(deltaSV_groups)[conf].mean()
    #conf_by_sv_certain_count = local_df.loc[certain_chosen].groupby(deltaSV_groups)[conf].size()
    #print(f"{conf_by_sv_certain_count}")
    conf_by_sv_certain_sem = local_df.loc[certain_chosen].groupby(deltaSV_groups)[conf].sem()
    
    plot_data(conf_by_sv.index, 
              [conf_by_sv, conf_by_sv_risky, conf_by_sv_certain],
              error_list = None, #[conf_by_sv_sem, conf_by_sv_risky_sem, conf_by_sv_certain_sem],
              labels=['all choices', 'chose risky option', 'chose certain option'], 
              colors=['black', 'red', 'blue'],
              title=f"Vevaiometric Curve ({conf_title}) for P(risky)={choice_title}",
              xlabel=f"{deltaSV_title}", 
              ylabel=f"{conf_title}",
              filename=rf"./plots (final)\deltaSV_vs_conf_analysis\{deltaSV_filename}_vs_{conf_filename}_{choice_filename}2.png")
    
def deltaSV_confidence_analysis():
    # get plots for deltaSV vs confidence based on choice type
    deltaSV_vs_confidence(df, "Confidence", "Subjective Value Difference", "All", "Confidence", "deltaSV", "all", "confidence") # all choice types
    deltaSV_vs_confidence(unsafe_df, "Confidence", "Subjective Value Difference", "0.5", "Confidence", "deltaSV", "0.5", "confidence") # unsafe choice type (P(risky)=0.5)
    deltaSV_vs_confidence(safe_df, "Confidence", "Subjective Value Difference", "0.9", "Confidence", "deltaSV", "0.9", "confidence") # safe choice type (P(risky)=0.9)

    # get plots for SPTdeltaSV vs confidence based on choice type
    deltaSV_vs_confidence(df, "Confidence", "SPT Subjective Value Difference", "All", "Confidence", "spt_deltaSV", "all", "confidence") # all choice types
    deltaSV_vs_confidence(unsafe_df, "Confidence", "SPT Subjective Value Difference", "0.5", "Confidence", "spt_deltaSV", "0.5", "confidence") # unsafe choice type (P(risky)=0.5)
    deltaSV_vs_confidence(safe_df, "Confidence", "SPT Subjective Value Difference", "0.9", "Confidence", "spt_deltaSV", "0.9", "confidence") # safe choice type (P(risky)=0.9)

    # get plots for deltaSV vs reaction time based on choice type
    deltaSV_vs_confidence(df, "rt", "Subjective Value Difference", "All", "Response Time", "deltaSV", "all", "RT") # all choice types
    deltaSV_vs_confidence(unsafe_df, "rt", "Subjective Value Difference", "0.5", "Response Time", "deltaSV", "0.5", "RT") # unsafe choice type (P(risky)=0.5)
    deltaSV_vs_confidence(safe_df, "rt", "Subjective Value Difference", "0.9", "Response Time", "deltaSV", "0.9", "RT") # safe choice type (P(risky)=0.9)




# run analyses
psychometric_analysis()
confidence_accuracy_analysis()
deltaSV_confidence_analysis()