import numpy as np
import nengo
import nengo_spa as spa

from .configurator import cfg
from .vocabulator import vocab
from .loggerator import logger
from .modules import Stimulus, Vision, ProdSys, RewardEval, InfoEnc
from .modules import TrfmSys, Memory, Monitor, InfoDec, Motor
from .modules import InstrStimulus, InstrProcess

# #### DEBUG DUMMY NETWORK IMPORTS ####
# from _spaun.modules.experimenter import StimulusDummy as Stimulus  # noqa
# from _spaun.modules.vision_system import VisionSystemDummy as Vision  # noqa
# from _spaun.modules.working_memory import WorkingMemoryDummy as Memory  # noqa
# from _spaun.modules.transform_system import TransformationSystemDummy as TrfmSys  # noqa
# from _spaun.modules.motor_system import MotorSystemDummy as Motor

# TODO: put in utils
class VocabWrapper:
    def __init__(self, vocab):
        self._vocab = vocab

    def __getattr__(self, name):
        # Delegate attribute access to vocab.parse
        return self._vocab.parse(name)

def Spaun(vocab):
    v = VocabWrapper(vocab.main)
    model = spa.Network(label='Spaun', seed=cfg.seed)
    with model:
        model.config[nengo.Ensemble].max_rates = cfg.max_rates
        model.config[nengo.Ensemble].neuron_type = cfg.neuron_type
        model.config[nengo.Connection].synapse = cfg.pstc

        if 'S' in cfg.spaun_modules:
            model.stim = Stimulus()
            model.instr_stim = InstrStimulus()
            model.monitor = Monitor()
        if 'V' in cfg.spaun_modules:
            model.vis = Vision()
            vis = model.vis.output
            vis_mem = model.vis.vis_main_mem.output
        if 'P' in cfg.spaun_modules:
            model.ps = ProdSys()
            ps_task_in = model.ps.task_mb.input
            ps_task_out = model.ps.task
            ps_state_in = model.ps.state_mb.input
            ps_state_out = model.ps.state
            ps_dec_in = model.ps.dec_mb.input
            ps_dec_out = model.ps.dec
            ps_action = model.ps.action_in
        if 'R' in cfg.spaun_modules:
            model.reward = RewardEval()
        if 'E' in cfg.spaun_modules:
            model.enc = InfoEnc()
        if 'W' in cfg.spaun_modules:
            model.mem = Memory()
        if 'T' in cfg.spaun_modules:
            model.trfm = TrfmSys()
            trfm_compare = model.trfm.compare
            trfm_input = model.trfm.select_out.input6
        if 'D' in cfg.spaun_modules:
            model.dec = InfoDec()
        if 'M' in cfg.spaun_modules:
            model.mtr = Motor()
        if 'I' in cfg.spaun_modules:
            model.instr = InstrProcess()
            instr_en = model.instr.enable_in_sp
            instr_task = model.instr.task_output
            instr_data = model.instr.output
            instr_state = model.instr.state_output
            instr_dec = model.instr.dec_output

        model.learn_conns = []

            

        if hasattr(model, 'vis') and hasattr(model, 'ps'):
            with spa.ActionSelection() as action_sel:
                spa.ifmax(0.5 * (spa.dot(ps_task_out, v.X) + spa.dot(vis, v.ZER)), 
                        v.W >> ps_task_in, v.TRANS0 >> ps_state_in, v.FWD >> ps_dec_in)
                spa.ifmax(spa.dot(ps_task_out, v.W-v.DEC) - spa.dot(vis, v.QM),
                        ps_state_out >> ps_state_in) # noqa
                # Copy drawing task format: A0[r]?X

                spa.ifmax(0.5 * (spa.dot(ps_task_out, v.X) + spa.dot(vis, v.ONE)),
                        v.R >> ps_task_in, v.TRANS0 >> ps_state_in, v.FWD >> ps_dec_in)
                spa.ifmax(spa.dot(ps_task_out, v.R-v.DEC) - spa.dot(vis, v.QM),
                        ps_state_out >> ps_state_in)
                # Digit recognition task format: A1[r]?X
                
                spa.ifmax(0.5 * (spa.dot(ps_task_out, v.X) + spa.dot(vis, v.TWO)) - spa.dot(vis, v.QM),
                        v.L >> ps_task_in, v.LEARN >> ps_state_in, v.FWD >> ps_dec_in)
                # Learning task format: A2?X<REWARD>?X<REWARD>?X<REWARD>?X<REWARD>?X<REWARD>    # noqa


                if hasattr(model, 'reward'):
                    for s in vocab.ps_action_learn_sp_strs:
                        spa.ifmax(0.5 * (spa.dot(ps_task_out, 2*v.L) - 1) - spa.dot(vis, v.QM),
                                v.s >> ps_action, v.LEARN >> ps_state_in, v.NONE >> ps_dec_in) 
                
                spa.ifmax(0.5 * (spa.dot(ps_task_out, v.X) + spa.dot(vis, v.THR)),
                        v.M >> ps_task_in, v.TRANS0 >> ps_state_in, v.FWD >> ps_dec_in)
                spa.ifmax(spa.dot(ps_task_out, v.M-v.DEC) - spa.dot(vis, v.F + v.R + v.QM),
                        ps_state_out >> ps_state_in)
                spa.ifmax(0.5 * (spa.dot(ps_task_out, v.M) + spa.dot(vis, v.F)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                        v.FWD >> ps_dec_in)
                spa.ifmax(0.5 * (spa.dot(ps_task_out, v.M) + spa.dot(vis, v.R)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                        v.REV >> ps_dec_in)
                # Working memory task format: A3[rr..rr]?XXX
                # Reverse recall task format: A3[rr..rr]R?XXX

                if hasattr(model, 'trfm'):
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.X) + spa.dot(vis, v.FOR)),
                                v.C >> ps_task_in, v.TRANS0 >> ps_state_in, v.FWD >> ps_dec_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.C) + spa.dot(ps_state_out, v.TRANS0)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                                v.CNT0 >> ps_state_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.C) + spa.dot(ps_state_out, v.CNT0)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                                v.CNT1 >> ps_state_in)
                    spa.ifmax((0.25 * (spa.dot(ps_task_out, v.DEC) + spa.dot(ps_state_out, v.CNT1)) + 0.5 * spa.dot(trfm_compare, v.NO_MATCH)) + # 'SemanticPointer' object has no attribute 'rdot'
                               (spa.dot(ps_dec_out, v.CNT) - 1) - spa.dot(vis, v.QM),
                                v.CNT >> ps_dec_in, v.CNT1 >> ps_state_in)
                    # Counting task format: A4[START_NUM][NUM_COUNT]?X..X

                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.X) + spa.dot(vis, v.FIV)),
                                v.A >> ps_task_in, v.TRANS0 >> ps_state_in, v.FWD >> ps_dec_in)
                    spa.ifmax(spa.dot(ps_task_out, v.A-v.DEC) - spa.dot(vis, v.K + v.P + v.QM),
                                ps_state_out >> ps_state_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.A) + spa.dot(vis, v.K)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                                v.QAK >> ps_state_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.A) + spa.dot(vis, v.P)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                                v.QAP >> ps_state_in)
                    # Question answering task format: A5[rr..rr]P[r]?X (probing item in position)           # noqa
                    #                                 A5[rr..rr]K[r]?X (probing position of item (kind))    # noqa

                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.X) + spa.dot(vis, v.SIX)),
                                v.V >> ps_task_in, v.TRANS0 >> ps_state_in, v.FWD >> ps_dec_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.V) + spa.dot(ps_state_out, v.TRANS0)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                                v.TRANS1 >> ps_state_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.V) + spa.dot(ps_state_out, v.TRANS1)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                                v.TRANS0 >> ps_state_in)
                    # Rapid variable creation task format: A6{[rr..rr][rr..rr]:NUM_EXAMPLES}?XX..XX     # noqa

                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.X) + spa.dot(vis, v.SEV)),
                                v.F >> ps_task_in, v.TRANS0 >> ps_state_in, v.FWD >> ps_dec_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.F) + spa.dot(ps_state_out, v.TRANS0)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                                v.TRANS1 >> ps_state_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.F) + spa.dot(ps_state_out, v.TRANS1)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                                v.TRANS2 >> ps_state_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.F) + spa.dot(ps_state_out, v.TRANS2)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                                v.TRANS0 >> ps_state_in)
                    # Fluid intelligence task format: A7[CELL1_1][CELL1_2][CELL1_3][CELL2_1][CELL2_2][CELL2_3][CELL3_1][CELL3_2]?XX..XX     # noqa


                    # Reaction task
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.X) + spa.dot(vis, v.EIG)),
                                v.REACT >> ps_task_in, v.DIRECT >> ps_state_in, v.FWD >> ps_dec_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.REACT) + spa.dot(vis_mem, v.ONE)),
                                v.POS1*v.THR >> trfm_input)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.REACT) + spa.dot(vis_mem, v.TWO)),
                                v.POS1*v.FOR >> trfm_input)
                    # Stimulus response (hardcoded reaction) task format: A8?1X<expected 3>?2X<expected 4>    # noqa

                    # Compare task -- See two items, check if their class matches each other # noqa
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.X) + spa.dot(vis, v.C)),
                                v.CMP >> ps_task_in, v.TRANS1 >> ps_state_in, v.FWD >> ps_dec_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.CMP) + spa.dot(ps_state_out, v.TRANS1)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                                v.TRANS2 >> ps_state_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.CMP) + spa.dot(ps_state_out, v.TRANS2)) - spa.dot(vis, v.QM) - spa.dot(ps_task_out, v.DEC),
                                v.TRANSC >> ps_state_in)
                    # List / item matching task format: AC[r][r]?X<expected 1 if match, 0 if not>                                                      # noqa
            
                if hasattr(model, 'trfm') and hasattr(model, 'instr'):
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.X) + spa.dot(vis, v.NIN)),
                                v.INSTR >> ps_task_in, v.DIRECT >> ps_state_in, v.FWD >> ps_dec_in)
                    spa.ifmax(spa.dot(ps_task_out, v.INSTR) - spa.dot(vis, v.QM + v.A + v.M + v.P + v.CLOSE) - spa.dot(ps_state_out, v.INSTRP),
                                v.ENABLE >> instr_en, instr_task >> ps_task_in, instr_state >> ps_state_in, instr_dec >> ps_dec_in, instr_data >> trfm_input)
                    spa.ifmax(1.5 * spa.dot(vis, v.M + v.V),
                                v.INSTR >> ps_task_in, v.TRANS0 >> ps_state_in, v.FWD >> ps_dec_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.INSTR) + spa.dot(vis, v.P)),
                                v.INSTR >> ps_task_in, v.INSTRP >> ps_state_in)
                    spa.ifmax(0.5 * (spa.dot(ps_task_out, v.INSTR) + spa.dot(ps_state_out, v.INSTRP)),
                                v.INSTR >> ps_task_in, v.TRANS0 >> ps_state_in)
                    # Instructed tasks task formats:
                    # Instructed stimulus response task format: A9?rX<expected answer from instruction>?rX<expected answer from instruction>    # noqa
                    # Instructed custom task format: M<0-9>[INSTRUCTED TASK FORMAT]?XX..XX                                                      # noqa
                    # Instructed positional task formats: MP<0-9>[INSTRUCTED TASK FORMAT]?XX..XX V[INSTRUCTED TASK FORMAT]?XX..XX               # noqa
                    #     - P<0-9> selects appropriate instruction from list of instructions                                                    # noqa
                    #     - V increments instruction position by 1

                spa.ifmax(spa.dot(vis, v.QM) - 0.6 * spa.dot(ps_task_out, v.W+v.C+v.V+v.F+v.L+v.REACT),
                            ps_task_out + v.DEC >> ps_task_in, ps_state_out + 0.5 * v.TRANS0 >> ps_state_in, ps_dec_out + 0.5 * v.FWD >> ps_dec_in)
                spa.ifmax(0.5 * (spa.dot(vis, v.QM) + spa.dot(ps_task_out, v.W-v.DEC)),
                            v.W + v.DEC >> ps_task_in, ps_state_out >> ps_state_in, v.DECW >> ps_dec_in)
                spa.ifmax(0.5 * (spa.dot(vis, v.QM) + spa.dot(ps_task_out, v.C-v.DEC)),
                            v.C + v.DEC >> ps_task_in, ps_state_out >> ps_state_in, v.CNT >> ps_dec_in)
                spa.ifmax(0.5 * (spa.dot(vis, v.QM) + spa.dot(ps_task_out, v.V+v.F-v.DEC)),
                            ps_task_out + v.DEC >> ps_task_in, ps_state_out >> ps_state_in, v.DECI >> ps_dec_in)
                spa.ifmax(0.7 * spa.dot(vis, v.QM) + 0.3 * spa.dot(ps_task_out, v.L),
                            v.L + v.DEC >> ps_task_in, v.LEARN >> ps_state_in, v.FWD >> ps_dec_in)
                spa.ifmax(0.5 * (spa.dot(vis, v.QM) + spa.dot(ps_task_out, v.REACT)),
                            v.REACT + v.DEC >> ps_task_in, v.DIRECT >> ps_state_in, v.FWD >> ps_dec_in)
                spa.ifmax(0.75 * spa.dot(ps_task_out, v.DEC-v.REACT-v.INSTR) - spa.dot(ps_state_out, v.LEARN)
                           - spa.dot(ps_dec_out, v.CNT) - spa.dot(vis, v.QM + v.A + v.M),
                            ps_task_out >> ps_task_in, ps_state_out >> ps_state_in, ps_dec_out >> ps_dec_in)
                

            # model.bg = action_sel.bg
            # model.thal = action_sel.thal

            # model.bg = spa.BasalGanglia(actions=actions, input_synapse=0.008,
            #                             label='Basal Ganglia')
            # model.thal = spa.Thalamus(model.bg, subdim_channel=1,
            #                         mutual_inhibit=1, route_inhibit=5.0,
            #                         label='Thalamus')

        # ----- Set up connections (and save record of modules) -----
        if hasattr(model, 'vis'):
            model.vis.setup_connections(model)
        if hasattr(model, 'ps'):
            model.ps.setup_connections(model)
            # Modify any 'channel' ensemble arrays to have
            # get_optimal_sp_radius radius sizes
            for net in model.ps.all_networks:
                if net.label is not None and net.label[:7] == 'channel': # 'channel' -- refers to intermediate states created by nengo.spa by actions. check if the nengo_spa does the same and if it has the same name
                    for ens in net.all_ensembles:
                        ens.radius = cfg.get_optimal_sp_radius()
        if hasattr(model, 'bg'):
            if hasattr(model, 'reward'):
                # Clear learning transforms
                # del cfg.learn_init_transforms[:]

                # with model.bg: # only for reward learning, which we are skipping
                #     # Generate random biases for each learn action, so that
                #     # there is some randomness to the initial action choice
                #     bias_node = nengo.Node(1)
                #     bias_ens = nengo.Ensemble(cfg.n_neurons_ens, 1,
                #                               label='BG Bias Ensemble')
                #     nengo.Connection(bias_node, bias_ens)

                #     for i in range(model.bg.input.size_in): ##?
                #         init_trfm = (np.random.random() *
                #                      cfg.learn_init_trfm_max)
                #         trfm_val = cfg.learn_init_trfm_bias + init_trfm
                #         model.learn_conns.append(
                #             nengo.Connection(bias_ens, model.bg.input[i],
                #                              transform=trfm_val))
                #         cfg.learn_init_transforms.append(trfm_val)
                logger.write("# learn_init_trfms: %s\n" %
                             (str(cfg.learn_init_transforms)))
        if hasattr(model, 'thal'):
            pass
        if hasattr(model, 'reward'):
            model.reward.setup_connections(model, model.learn_conns)
        if hasattr(model, 'enc'):
            model.enc.setup_connections(model)
        if hasattr(model, 'mem'):
            model.mem.setup_connections(model)
        if hasattr(model, 'trfm'):
            model.trfm.setup_connections(model)
            # Modify any 'channel' ensemble arrays to have
            # get_optimal_sp_radius radius sizes
            for net in model.trfm.all_networks:
                if net.label is not None and net.label[:7] == 'channel': # 'channel' -- refers to intermediate states created by nengo.spa by actions
                    for ens in net.all_ensembles:
                        ens.radius = cfg.get_optimal_sp_radius()
        if hasattr(model, 'dec'):
            model.dec.setup_connections(model)
        if hasattr(model, 'mtr'):
            model.mtr.setup_connections(model)
        if hasattr(model, 'instr'):
            model.instr.setup_connections(model)
        if hasattr(model, 'monitor'):
            model.monitor.setup_connections(model)

    return model
