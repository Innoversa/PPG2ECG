import seq2seq_utils as zu
import sys
import numpy as np
from pprint import pprint
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pdb


def form_dict(bp_key, fig, rmse, mae, mean, std, rval, pval):
    """form_dict
        Make dictionary from bland_altman and other statistics

    Arguments:
        bp_key {str} -- whether ABP, SBP, or DBP, used to make key for dictionary
        fig {pyplot_figure} -- bland altman plot
        rmse {float} -- Rooted Mean Squared Error, the lower the better
        mae {float} -- Mean Absolute Error, the lower the better
        rval {float} -- Pearon's R value, between 0~1, the higher the better

    Returns:
        {dictionary} -- dictionary containing figure and metrics of given bp_key
    """
    return {
        f"{bp_key}_RMSE": rmse,
        f"{bp_key}_MAE": mae,
        f"{bp_key}_MEAN": mean,
        f"{bp_key}_STD": std,
        f"{bp_key}_Pearson": rval,
        f"{bp_key}_Bland_Altman": fig,
    }


# Plot waveform
def plot_waveform(
    pred_arr,
    test_arr,
    freq,
    title="default title",
    x_lab="default_x",
    y_lab="default_y",
    plot_style=".-.",
    add_metrics=False,
):
    """plot_waveform
        Plotting a comparison plot between pred and test

    Arguments:
        pred_arr {list} -- Predicted waveform sequence of given ecg_type
        test_arr {list} -- Ground Truth waveform sequence of given ecg_type
        freq {int} -- frequency of the device (in Hz, MIMIC ABP has 125 Hz),
                            setting 0 then X-label will be in sample domain, otherwise in time domain

    Keyword Arguments:
        title {str} -- Custom title for the given plot, usually containing info
                            about model and patient involved in the given prediction (default: {"Default Title"})
        x_lab {str} -- name of X label (default: {"default_x"})
        y_lab {str} -- name of y label (default: {"default_y"})
        plot_style {str} -- Style of the line or dot (default: {".-."})
        add_metrics {bool} -- Whether to include metrics info on the plot
                                (metrics include RMSE, MAE, and R) (default: {False})

    Returns:
        _type_ -- _description_
    """
    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots()
    ax.set_title(title)
    x_axis = np.arange(len(pred_arr)) / freq
    ax.plot(x_axis, pred_arr, plot_style, label="pred", lw=2, alpha=0.6)
    ax.plot(x_axis, test_arr, plot_style, label="test", lw=2, alpha=0.6)
    # ax.set_ylim(20, 180)
    if add_metrics:
        rmse, mae, mean, std, rval, pval = zu.calc_metrics(pred=pred_arr, test=test_arr)
        local_metrics = [
            f"RMSE={rmse:.3f}",
            f"MAE={mae:.3f}",
            f"Pearson={rval:.3f}",
        ]
        for i, t in enumerate(local_metrics):
            x_pos = int(x_axis[-1] * 0.95)
            y_pos = 1 - (i * 0.1)
            ax.text(x_pos, y_pos, t, color="k", fontsize=10)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.legend(loc=1)
    fig.set_size_inches(10, 7)
    fig.set_dpi(75)
    plt.close()
    return fig, ax


def plot_three_waveform(
    pred_arr_list,
    test_arr_list,
    freq,
    title="default title",
    x_lab="default_x",
    y_lab="default_y",
    plot_style=".-.",
    add_metrics=False,
):
    """plot_three_waveforms
        Plotting 3 plots between pred and test

    Arguments:
        pred_arr_list {list} -- lists of Predicted waveform sequence of given ecg_type
        test_arr_list {list} -- lists of Ground Truth waveform sequence of given ecg_type
        freq {int} -- frequency of the device (in Hz, MIMIC ABP has 125 Hz),
                            setting 0 then X-label will be in sample domain, otherwise in time domain

    Keyword Arguments:
        title {str} -- Custom title for the given plot, usually containing info
                            about model and patient involved in the given prediction (default: {"Default Title"})
        x_lab {str} -- name of X label (default: {"default_x"})
        y_lab {str} -- name of y label (default: {"default_y"})
        plot_style {str} -- Style of the line or dot (default: {".-."})
        add_metrics {bool} -- Whether to include metrics info on the plot
                                (metrics include RMSE, MAE, and R) (default: {False})

    Returns:
        fig -- pyplot figure
        ax  -- pyplot axis
    """
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(len(pred_arr_list), 1, sharex=True, sharey=True)
    fig.suptitle(title, fontdict={"weight": "black"})
    for i, pred_arr in enumerate(pred_arr_list):
        x_axis = np.arange(len(pred_arr)) / freq
        ax[i].plot(x_axis, pred_arr, plot_style, label="pred", lw=1.5, alpha=0.6)
        ax[i].plot(
            x_axis, test_arr_list[i], plot_style, label="test", lw=1.5, alpha=0.6
        )
        # ax[i].set_ylim(20, 180)
        if add_metrics:
            rmse, mae, mean, std, rval, pval = zu.calc_metrics(
                pred=pred_arr, test=test_arr_list[i]
            )
            local_metrics = [
                f"RMSE={rmse:.3f}",
                f"MAE={mae:.3f}",
                f"Pearson={rval:.3f}",
                f"p-value={pval:.3f}",
                f"mean={mean:.3f}",
                f"std={std:.3f}",
            ]
            for j, t in enumerate(local_metrics):
                x_pos = 0 - x_axis[int(len(x_axis) * 0.055)]
                y_pos = 180 - (j * 15)
                ax[i].text(
                    x_pos,
                    y_pos,
                    t,
                    color="k",
                    fontdict={
                        "size": 10,
                        "family": "monospace",
                    },
                )
    fig.supxlabel(x_lab)
    fig.supylabel(y_lab)
    ax[0].legend(loc=1)
    fig.set_size_inches(18, 9)
    fig.set_dpi(75)
    plt.close()
    return fig, ax


class PPG2ECG_Visual:
    """PPG2ECG_Visual
    A Visualization class dedicated to provide figures and metrics for MIMIC Arterial Blood Pressure waveform Prediction Tasks
    """

    def __init__(
        self,
        waveform_dict,
        patient_name=None,
        model_name=None,
        sampled_freq=125,
        use_wandb=False,
    ):
        """__init__ Initialize the PPG2ECG_Visual class with
            waveform data; patient information, and model information

        Arguments:
            waveform_dict {dict} -- Dictionary of both predicted and ground truth waveform of
                Arterial Blood Pressure and its extracted sequence of Systolic and Diastolic Blood Pressures

        Keyword Arguments:
            patient_name {str} -- Name of the MIMIC Patient which the waveform belong to (default: {None})
            model_name {str} -- Name/Type of the Model that produced the predicted waveform (default: {None})
            sampled_freq {int} -- Sampling frequency of the waveform in Hertz, MIMIC III default is 125Hz (default: {125})
            use_wandb {bool} -- Whether to produce Weights & Bias Image or default Pyplot Figure (default: {False})

        """
        self.pred_dict = {"ECG": waveform_dict["ecg_pred"]}
        self.test_dict = {"ECG": waveform_dict["ecg_test"]}
        self.patient_name = patient_name
        self.model_name = model_name
        self.freq = sampled_freq
        self.bp_dict = {"ECG": "Lead II ECG"}
        self.use_wandb = use_wandb
        self.PPG2ECG_Visual_dict = {}

    def fig_to_wandb_image(self, fig, caption="default caption"):
        """fig_to_wandb_image Whether to convert pyplot figures to
            Weight & Bias Image (https://docs.wandb.ai/guides/track/log/media)

        Arguments:
            fig {pyplot} -- pyplot figure to be converted

        Keyword Arguments:
            caption {str} -- caption of the Weight & Bias Image (default: {"default caption"})

        Returns:
            wandb Image -- outputted Weights & Bias Image
        """
        if self.use_wandb:
            return wandb.Image(fig, caption)
        return fig

    def plot_bland_altman(self, ecg_type):
        """plot_bland_altman
            Formulating inputs based on object variable;
            Calling the external function;
            Returning the constructed dictionary of information

        Arguments:
            ecg_type {str} -- whether to construct input for ABP, SBP, or DBP

        Returns:
            dict -- returning a dictionary containing desired visuliazation
        """
        bp_key = ecg_type.upper()
        fig, rmse, mae, mean, std, rval, pval = plot_bland_altman(
            pred_arr=self.pred_dict[bp_key],
            test_arr=self.test_dict[bp_key],
            title=f"{self.bp_dict[bp_key]} Bland Altman for\nPatient: {self.patient_name}",
            ecg_type=bp_key,
            return_plt=True,
        )
        fig = self.fig_to_wandb_image(
            fig, caption=f"{self.bp_dict[bp_key]}_Bland_Altman"
        )
        return form_dict(bp_key, fig, rmse, mae, mean, std, rval, pval)

    def plot_Prediction(
        self, ecg_type, vis_st_idx=0, vis_seq_len=1250, add_metrics=False
    ):
        """plot_Prediction
            Formulating inputs based on object variable;
            Calling the external function;
            Returning the constructed dictionary of information

        Arguments:
            ecg_type {str} -- whether to construct input for ECG, SBP, or DBP

        Keyword Arguments:
            vis_st_idx {int} -- starting index of the sequence of the visualization (default: {0})
            vis_seq_len {int} -- length of sequence of the visualization (default: {1250})
            add_metrics {bool} -- whether to include metrics info on the visualization (default: {False})

        Returns:
            dict -- returning a dictionary containing desired visuliazation
        """
        bp_key = ecg_type.upper()
        x_lab = "Time (seconds)" if bp_key == "ECG" else "Data Points (# of Samples)"
        freq = self.freq if bp_key == "ECG" else 1
        st_idx = max(0, vis_st_idx)
        ed_idx = min(len(self.pred_dict[bp_key]), vis_st_idx + vis_seq_len)
        print(
            f"st_idx: {st_idx}, ed_idx: {ed_idx}, print_len = {len(self.test_dict[bp_key])}"
        )
        wave_fig, ax = plot_waveform(
            pred_arr=self.pred_dict[bp_key][st_idx:ed_idx],
            test_arr=self.test_dict[bp_key][st_idx:ed_idx],
            freq=freq,  # Match frequency for A, else point by point
            x_lab=x_lab,
            y_lab=f"{self.bp_dict[bp_key]} (mmHg)",
            title=f"{self.model_name.upper()} {self.bp_dict[bp_key]} Prediction for\nPatient: {self.patient_name}",
            add_metrics=add_metrics,
        )
        wave_fig = self.fig_to_wandb_image(
            wave_fig, caption=f"{self.bp_dict[bp_key]}_Prediction"
        )
        return {f"{bp_key}_Waveform": wave_fig}

    def plot_three_Predictions(
        self, ecg_type, vis_st_idx=0, vis_seq_len=1250, add_metrics=False
    ):
        """plot_Prediction
            Formulating inputs based on object variable;
            Calling the external function;
            Returning the constructed dictionary of information

        Arguments:
            ecg_type {str} -- whether to construct input for ABP, SBP, or DBP

        Keyword Arguments:
            vis_st_idx {int} -- starting index of the sequence of the visualization (default: {0})
            vis_seq_len {int} -- length of sequence of the visualization (default: {250})
            add_metrics {bool} -- whether to include metrics info on the visualization (default: {False})

        Returns:
            dict -- returning a dictionary containing desired visuliazation
        """
        bp_key = ecg_type.upper()
        x_lab = "Time (seconds)" if bp_key == "ABP" else "Data Points (# of Samples)"
        freq = self.freq if bp_key == "ABP" else 1
        pred_list = []
        test_list = []
        vis_seq_len = min(vis_seq_len, int(len(self.pred_dict[bp_key]) / 3))
        for i in range(3):
            st_idx = max(0, vis_st_idx) + i * vis_seq_len
            ed_idx = (
                min(len(self.pred_dict[bp_key]), vis_st_idx + vis_seq_len)
                + i * vis_seq_len
            )
            pred_list.append(self.pred_dict[bp_key][st_idx:ed_idx])
            test_list.append(self.test_dict[bp_key][st_idx:ed_idx])
            print(f"st_idx: {st_idx}, ed_idx: {ed_idx}")
        wave_fig, ax = plot_three_waveform(
            pred_arr_list=pred_list,
            test_arr_list=test_list,
            freq=freq,  # Match frequency for A, else point by point
            x_lab=x_lab,
            y_lab=f"{self.bp_dict[bp_key]} (mmHg)",
            title=f"{self.model_name.upper()} {self.bp_dict[bp_key]} Prediction for\nPatient: {self.patient_name}",
            add_metrics=add_metrics,
        )
        wave_fig = self.fig_to_wandb_image(
            wave_fig, caption=f"{self.bp_dict[bp_key]}_Prediction"
        )
        return {f"{bp_key}_Three_Waveform": wave_fig}

    def plot_confusion_matrix(self, ecg_type):
        """plot_confusion_matrix
            Formulating inputs based on object variable;
            Calling the external function;
            Returning the constructed dictionary of information

        Arguments:
            ecg_type {str} -- whether to construct input for ABP, SBP, or DBP

        Returns:
            dict -- returning a dictionary containing desired visuliazation
        """
        bp_key = ecg_type.upper()
        cf_fig = confusion_matrix_of_stages(
            pred_arr=self.pred_dict[bp_key],
            test_arr=self.test_dict[bp_key],
            ecg_type=bp_key,
            pname=f"{self.patient_name}",
        )
        cf_fig = self.fig_to_wandb_image(
            cf_fig, caption=f"{self.bp_dict[bp_key]}_Confusion_Matrix"
        )
        return {f"{bp_key}_Confusion_Matrix": cf_fig}

    def plot_everything(self):
        """plot_everything
            Plotting all of Bland Altman, Waveform, and Confusion Matrix Plots
            Returning a dictionary of all information this object does

        Returns:
            dict -- A dictionary of all MIMIC Visualization information available
        """
        # adding plot_waveform
        self.PPG2ECG_Visual_dict.update(self.plot_Prediction("ecg", add_metrics=True))
        # self.PPG2ECG_Visual_dict.update(self.plot_Prediction("sbp", add_metrics=True))
        # self.PPG2ECG_Visual_dict.update(self.plot_Prediction("dbp", add_metrics=True))
        self.PPG2ECG_Visual_dict.update(
            self.plot_three_Predictions("ecg", add_metrics=True)
        )
        # adding bland altman
        # self.PPG2ECG_Visual_dict.update(self.plot_bland_altman("sbp"))
        # self.PPG2ECG_Visual_dict.update(self.plot_bland_altman("dbp"))
        # adding confusion matrix
        # self.PPG2ECG_Visual_dict.update(self.plot_confusion_matrix("sbp"))
        # self.PPG2ECG_Visual_dict.update(self.plot_confusion_matrix("dbp"))
        return self.PPG2ECG_Visual_dict


if __name__ == "__main__":
    # Declaring random numbers for testing purposes
    use_wandb = True
    import wandb

    if use_wandb:
        wandb.init(
            project="play_ground",
            reinit=True,
            tags=["visual combine"],
        )
        log_dict = {}
    np.random.seed(12)
    wf_dict = {
        "abp_pred": np.random.normal(100, 20, 4000),
        "abp_test": np.random.normal(100, 20, 4000),
        "sbp_pred": np.random.normal(120, 20, 375),
        "sbp_test": np.random.normal(120, 20, 375),
        "dbp_pred": np.random.normal(80, 20, 375),
        "dbp_test": np.random.normal(80, 20, 375),
    }
    # declaring Visual Class
    MV = PPG2ECG_Visual(
        wf_dict,
        patient_name="Sicong(clearloveyanzhen)",
        model_name="Deep Learning",
        use_wandb=use_wandb,
    )
    overall_visual_dict = MV.plot_everything()
    # testing visualization
    if not use_wandb:
        overall_visual_dict["DBP_Bland_Altman"].savefig(
            "../plot_dir/DBP_bland_altman.png"
        )
        overall_visual_dict["SBP_Bland_Altman"].savefig(
            "../plot_dir/SBP_bland_altman.png"
        )
        overall_visual_dict["SBP_Confusion_Matrix"].savefig(
            "../plot_dir/SBP_confusion_matrix.png"
        )
        overall_visual_dict["DBP_Confusion_Matrix"].savefig(
            "../plot_dir/DBP_confusion_matrix.png"
        )
        overall_visual_dict["ABP_Waveform"].savefig("../plot_dir/ABP_visual_test.png")
        overall_visual_dict["ABP_Three_Waveform"].savefig(
            "../plot_dir/ABP_Three_visual_test.png"
        )
        overall_visual_dict["SBP_Waveform"].savefig("../plot_dir/SBP_visual_test.png")
        overall_visual_dict["DBP_Waveform"].savefig("../plot_dir/DBP_visual_test.png")
    else:
        log_dict = overall_visual_dict
        wandb.log(log_dict)
    for each in overall_visual_dict:
        print(f"{each}\t{overall_visual_dict[each]}\t{type(overall_visual_dict[each])}")
    pprint(overall_visual_dict)
    print("This main func is used for testing purpose only")
