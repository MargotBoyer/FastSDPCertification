def objective_Lan(self):
   
    print("Adding objective function for Lan")
    if self.LAST_LAYER:
        print("LAST LAYER IN Lan")
        self.handler.Objective.add_linear_variable(
            "z",
            1,
            layer=self.K,
            neuron=self.ytrue,
            front_of_matrix=False,
        )
        self.handler.Objective.add_linear_variable(
            "z",
            -1,
            layer=self.K,
            neuron=self.ytarget,
            front_of_matrix=False,
        )

    else:
        print("NOT LAST LAYER IN Lan")
        for i in range(self.n[self.K - 1]):
            if (self.K - 1, i) in self.stable_inactives_neurons:
                continue
            self.handler.Objective.add_linear_variable(
                "z",
                self.network.W[self.K - 1][self.ytrue][i]
                - self.network.W[self.K - 1][self.ytarget][i],
                layer=self.K - 1,
                neuron=i,
                front_of_matrix=False,
            )
        self.handler.Objective.add_constant(
            value=self.network.b[self.K - 1][self.ytrue]
            - self.network.b[self.K - 1][self.ytarget]
        )


def objective_Md(self):
    if not self.use_active_neurons:
        assert self.keep_penultimate_actives , "keep_penultimate_actives must be True for Md objective function"
    if self.LAST_LAYER:
        self.handler.Objective.add_linear_variable(
            "z",
            1,
            layer=self.K,
            neuron=self.ytrue,
        )
        for j in self.ytargets:
            if j == self.ytrue:
                continue
            self.handler.Objective.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="z",
                layer=self.K,
                neuron=j,
                value=-1,
                front_of_matrix2=False,
            )
    else:
        for i in range(self.n[self.K - 1]):
            if (self.K - 1, i) in self.stable_inactives_neurons:
                continue
            else:
                self.handler.Objective.add_linear_variable(
                    "z",
                    value=self.network.W[self.K - 1][self.ytrue][i],
                    layer=self.K - 1,
                    neuron=i,
                    front_of_matrix=False,
                )

        self.handler.Objective.add_constant(
            value=self.network.b[self.K - 1][self.ytrue]
        )
        for j in self.ytargets:
            if j == self.ytrue:
                continue
            self.handler.Objective.add_linear_variable(
                "beta",
                class_label=j,
                value=-self.b[self.K - 1][j],
            )
            for i in range(self.n[self.K - 1]):
                if (self.K - 1, i) in self.stable_inactives_neurons:
                    continue
                # elif (self.K - 1, i) in self.stable_actives_neurons:

                #     self.handler.Objective.add_quad_variable_bounding(
                #         value=self.network.W[self.K - 1][j][i],
                #         layer=self.K - 1,
                #         neuron=i,
                #         class_label=j,
                #         alpha_1=self.alpha_1,
                #         alpha_2=self.alpha_2,
                #         type="lower",
                #     )
                else:
                    self.handler.Objective.add_quad_variable(
                        var1="beta",
                        class_label=j,
                        var2="z",
                        layer2=self.K - 1,
                        neuron2=i,
                        value=-self.network.W[self.K - 1][j][i],
                        front_of_matrix2=False,
                    )


def objective_Mzbar(self):
    if self.LAST_LAYER:
        self.handler.Objective.add_linear_variable(
            "z",
            1,
            layer=self.K,
            neuron=self.ytrue,
        )
        self.handler.Objective.add_linear_variable(
            "zbar",
            -1,
        )
    else:
        for i in range(self.n[self.K - 1]):
            if (self.K - 1, i) in self.stable_inactives_neurons:
                continue
            self.handler.Objective.add_linear_variable(
                "z",
                self.network.W[self.K - 1][self.ytrue][i],
                layer=self.K - 1,
                neuron=i,
            )

        self.handler.Objective.add_constant(
            value=self.network.b[self.K - 1][self.ytrue]
        )

        self.handler.Objective.add_linear_variable("zbar", -1)
