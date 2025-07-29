"""Implementation of action syntax."""

from collections import OrderedDict
from collections.abc import Mapping

import nengo

from nengo_spa.ast import dynamic
from nengo_spa.connectors import ModuleInput, RoutedConnection, input_vocab_registry
from nengo_spa.exceptions import SpaActionSelectionError, SpaTypeError
from nengo_spa.types import TScalar
import nengo_spa as spa


class ActionSelection(spa.ActionSelection):
    """
    Implements an action selection system with basal ganglia and thalamus.

    The *ActionSelection* instance has to be used as context manager and each
    potential action is defined by an `.ifmax` call providing an expression
    for the utility value and any number of effects (routing of information)
    to activate when this utility value is highest of all.

    Attributes
    ----------
    active : ActionSelection
        Class attribute providing the currently active ActionSelection
        instance (if any).
    built : bool
        Indicates whether the action selection system has been built
        successfully.
    bg : nengo.Network
        Basal ganglia network. Available after the action selection system has
        been built.
    thalamus : nengo.Network
        Thalamus network. Available after the action selection system has
        been built.

    See Also
    --------

    nengo_spa.modules.BasalGanglia : Default basal ganglia network
    nengo_spa.modules.Thalamus : Default thalamus network

    Examples
    --------

    .. code-block:: python

        with ActionSelection():
            ifmax(dot(state, sym.A), sym.B >> state)
            ifmax(dot(state, sym.B), sym.C >> state)
            ifmax(dot(state, sym.C), sym.A >> state)

    This will route the *B* Semantic Pointer to *state* when *state* is more
    similar to *A* than any of the other Semantic Pointers. Similarly, *C*
    will be routed to *state* when *state* is *B*. Once, *state* is *C*, it
    will be reset to *A* and the cycle begins anew.

    Further action selection examples:

      * :ref:`/examples/question-control.ipynb`
      * :ref:`/examples/spa-parser.ipynb`
      * :ref:`/examples/spa-sequence.ipynb`
      * :ref:`/examples/spa-sequence-routed.ipynb`
    """

    active = None

    def __init__(self, bg=None, thalamus=None, 
                 bg_config=None, thal_config=None, channel_config=None):
        super(ActionSelection, self).__init__()
        self.bg = bg
        self.thalamus = thalamus
        self.bg_config = bg_config or {}
        self.thal_config = thal_config or {}
        self.channel_config = channel_config or {}


    def _build(self):
        try:
            if len(RoutedConnection.free_floating) > 0:
                raise SpaActionSelectionError(
                    "All actions in an action selection context must be part "
                    "of an ifmax call."
                )
        finally:
            RoutedConnection.free_floating.clear()

        if len(self._utilities) <= 0:
            return

        self.bias = nengo.Node(1.0, label="bias")
        if self.bg is None:
            self.bg = dynamic.BasalGangliaRealization(len(self._utilities), **self.bg_config)
        if self.thalamus is None:
            self.thalamus = dynamic.ThalamusRealization(len(self._utilities), **self.thal_config)
        self.thalamus.connect_bg(self.bg)

        for index, utility in enumerate(self._utilities):
            self.bg.connect_input(utility, index=index)

        for index, action in enumerate(self._actions):
            for effect in action:
                if effect.fixed:
                    self.thalamus.connect_fixed(
                        index, effect.sink.input, transform=effect.transform()
                    )
                else:
                    self.thalamus.construct_gate(index, self.bias)
                    channel = self.thalamus.construct_channel(
                        effect.sink.input, effect.type, **self.channel_config
                    )
                    effect.connect_to(channel.input)
                    self.thalamus.connect_gate(index, channel)
        self.built = True

