import qutip as q
import numpy as np
import matplotlib.pyplot as plt


def plot_pops(t_list, new_states, pulse_time=None):

    t_list = 1e6 * np.array(t_list)

    C13_dms = [q.ptrace(new_state, 1) for new_state in new_states]
    NV_dms = [q.ptrace(new_state, 0) for new_state in new_states]

    fig, ax = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))

    ax[0].plot(
        t_list, [NV_dm.diag()[0].real for NV_dm in NV_dms], label=r"$m_s=|0\rangle$"
    )
    ax[0].plot(
        t_list, [NV_dm.diag()[1].real for NV_dm in NV_dms], label=r"$m_s=|-1\rangle$"
    )
    ax[1].plot(
        t_list,
        [C13_dm.diag()[0].real for C13_dm in C13_dms],
        label=r"$m_s=|+\frac{1}{2}\rangle$",
    )
    ax[1].plot(
        t_list,
        [C13_dm.diag()[1].real for C13_dm in C13_dms],
        label=r"$m_s=|-\frac{1}{2}\rangle$",
    )

    # plot settings
    ax[0].set_ylabel("Population")
    ax[0].set_title("NV center spin")
    ax[1].set_title("C13 spin")
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel(r"Time [$\mu$s]")
    ax[1].set_xlabel(r"Time [$\mu$s]")
    ax[0].set_ylim(0, 1.02)
    ax[1].set_ylim(0, 1.02)

    if pulse_time is not None:
        pulse_time *= 1e6
        y1 = np.full_like(t_list, 0)
        y2 = np.full_like(t_list, 1)
        ax[0].axvline(pulse_time, color="black", linestyle="--", label=r"$t_e$")
        ax[1].axvline(pulse_time, color="black", linestyle="--", label=r"$t_e$")
        ax[0].fill_between(
            t_list,
            y1,
            y2,
            where=(t_list >= 0) & (t_list <= pulse_time),
            color="lightgrey",
            alpha=0.5,
        )
        ax[1].fill_between(
            t_list,
            y1,
            y2,
            where=(t_list >= 0) & (t_list <= pulse_time),
            color="lightgrey",
            alpha=0.5,
        )

    return fig, ax


def plot_exp_values(t_list, new_states, pulse_time=None):

    t_list = 1e6 * np.array(t_list)

    C13_dms = [q.ptrace(new_state, 1) for new_state in new_states]
    NV_dms = [q.ptrace(new_state, 0) for new_state in new_states]

    fig, ax = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))

    ax[0].plot(
        t_list,
        [q.expect(q.sigmax(), NV_dm).real for NV_dm in NV_dms],
        label=r"$\langle S_x \rangle_{NV}$",
    )
    ax[0].plot(
        t_list,
        [q.expect(q.sigmax(), C13_dm).real for C13_dm in C13_dms],
        label=r"$\langle S_x \rangle_{C13}$",
    )
    ax[1].plot(
        t_list,
        [q.expect(q.sigmaz(), NV_dm).real for NV_dm in NV_dms],
        label=r"$\langle S_z \rangle_{NV}$",
    )
    ax[1].plot(
        t_list,
        [q.expect(q.sigmaz(), C13_dm).real for C13_dm in C13_dms],
        label=r"$\langle S_z \rangle_{C13}$",
    )

    # plot settings
    ax[0].set_ylabel("Expectation Value")
    ax[0].set_title("Coherences")
    ax[1].set_title("Populations")
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel(r"Time [$\mu$s]")
    ax[1].set_xlabel(r"Time [$\mu$s]")
    # ax[0].set_ylim(-1.02, 1.02) # for coherence
    ax[1].set_ylim(-1.02, 1.02)

    if pulse_time is not None:
        pulse_time *= 1e6
        y1 = np.full_like(t_list, -1)
        y2 = np.full_like(t_list, 1)
        ax[0].axvline(pulse_time, color="black", linestyle="--", label=r"$t_e$")
        ax[1].axvline(pulse_time, color="black", linestyle="--", label=r"$t_e$")
        ax[0].fill_between(
            t_list,
            y1,
            y2,
            where=(t_list >= 0) & (t_list <= pulse_time),
            color="lightgrey",
            alpha=0.5,
        )
        ax[1].fill_between(
            t_list,
            y1,
            y2,
            where=(t_list >= 0) & (t_list <= pulse_time),
            color="lightgrey",
            alpha=0.5,
        )

    return fig, ax


def plot_fids(t_list, fidelities, pulse_time=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    t_list = 1e6 * np.array(t_list)
    ax.plot(t_list, fidelities)

    # plot settings
    ax.set_ylabel("Fidelity")
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_ylim(0, 1.02)

    if pulse_time is not None:
        pulse_time *= 1e6
        y1 = np.full_like(t_list, 0)
        y2 = np.full_like(t_list, 1)
        ax.axvline(pulse_time, color="black", linestyle="--", label=r"$t_e$")
        ax.fill_between(
            t_list,
            y1,
            y2,
            where=(t_list >= 0) & (t_list <= pulse_time),
            color="lightgrey",
            alpha=0.5,
        )

    return fig, ax


def plot_log_negativity(t_list, log_negativities, pulse_time=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    t_list = 1e6 * np.array(t_list)
    ax.plot(t_list, log_negativities)

    # plot settings
    ax.set_ylabel("Logarithmic Negativity $E_N$")
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_ylim(0, 1.02)

    if pulse_time is not None:
        pulse_time *= 1e6
        y1 = np.full_like(t_list, 0)
        y2 = np.full_like(t_list, 1)
        ax.axvline(pulse_time, color="black", linestyle="--", label=r"$t_e$")
        ax.fill_between(
            t_list,
            y1,
            y2,
            where=(t_list >= 0) & (t_list <= pulse_time),
            color="lightgrey",
            alpha=0.5,
        )

    return fig, ax
