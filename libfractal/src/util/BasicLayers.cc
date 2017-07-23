/*
   Copyright 2015 Kyuyeon Hwang (kyuyeon.hwang@gmail.com)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 */


#include "BasicLayers.h"

#include <string>

#include "../core/Rnn.h"

namespace fractal
{

    namespace basicLayers
    {
        void AddLstmLayer(Rnn &rnn, const std::string name, const std::string biasLayer, const unsigned long delayAmount, const unsigned long size, const InitWeightParam &initWeightParam, int M_in,int quant_bit , const FLOAT initForgetGateBias)
        {
            const std::string prefix = name + ".";                                    
            InitWeightParam initForgetGateBiasParam = initWeightParam;

            LayerParam statePenaltyParam;
            //statePenaltyParam.statePenalty = 1e-6;

            initForgetGateBiasParam.mean = initForgetGateBias;
                LayerSpec layerspec;
                layerspec.dimX = size;
                layerspec.dimY = 1;
                layerspec.numMaps =1;
                layerspec.quant_bit = quant_bit;
            rnn.AddLayer(prefix + "INPUT", ACT_TANH, AGG_SUM, size,layerspec);                  
            rnn.AddLayer(prefix + "INPUT_GATE_PEEP", ACT_LINEAR, AGG_MULT, size,layerspec);     
            rnn.AddLayer(prefix + "FORGET_GATE_PEEP", ACT_LINEAR, AGG_MULT, size,layerspec);    
            rnn.AddLayer(prefix + "OUTPUT_GATE_PEEP", ACT_LINEAR, AGG_MULT, size,layerspec);    
            rnn.AddLayer(prefix + "INPUT_GATE", ACT_SIGMOID, AGG_SUM, size,layerspec);          
            rnn.AddLayer(prefix + "INPUT_GATE_MULT", ACT_LINEAR, AGG_MULT, size,layerspec);
            rnn.AddLayer(prefix + "MEMORY_CELL", ACT_LINEAR, AGG_SUM, size,layerspec, statePenaltyParam);
            rnn.AddLayer(prefix + "MEMORY_CELL_DELAY", ACT_LINEAR, AGG_MULT, size,layerspec);
            rnn.AddLayer(prefix + "FORGET_GATE", ACT_SIGMOID, AGG_SUM, size,layerspec);
            rnn.AddLayer(prefix + "FORGET_GATE_MULT", ACT_LINEAR, AGG_MULT, size,layerspec);
            rnn.AddLayer(prefix + "OUTPUT_SQUASH", ACT_TANH, AGG_SUM, size,layerspec);
            rnn.AddLayer(prefix + "OUTPUT_GATE", ACT_SIGMOID, AGG_SUM, size,layerspec);
            rnn.AddLayer(prefix + "OUTPUT", ACT_LINEAR, AGG_MULT, size,layerspec);
            rnn.AddLayer(prefix + "OUTPUT_DELAY", ACT_LINEAR, AGG_MULT, size,layerspec);

            ConnSpec connSpec;
            connSpec.connType = CONN_FULL;
            connSpec.M = M_in;
            rnn.AddConnection(prefix + "INPUT", prefix + "INPUT_GATE_MULT", 0, true,connSpec);
            rnn.AddConnection(prefix + "INPUT_GATE", prefix + "INPUT_GATE_MULT", 0, true,connSpec);
            rnn.AddConnection(prefix + "INPUT_GATE_MULT", prefix + "MEMORY_CELL", 0, true,connSpec);
            rnn.AddConnection(prefix + "MEMORY_CELL", prefix + "MEMORY_CELL_DELAY", delayAmount, true,connSpec, initWeightParam);
            rnn.AddConnection(prefix + "MEMORY_CELL_DELAY", prefix + "FORGET_GATE_MULT", 0, true,connSpec);
            rnn.AddConnection(prefix + "FORGET_GATE", prefix + "FORGET_GATE_MULT", 0, true,connSpec);
            rnn.AddConnection(prefix + "FORGET_GATE_MULT", prefix + "MEMORY_CELL", 0, true,connSpec);
            rnn.AddConnection(prefix + "MEMORY_CELL", prefix + "OUTPUT_SQUASH", 0, true,connSpec);
            rnn.AddConnection(prefix + "OUTPUT_SQUASH", prefix + "OUTPUT", 0, true,connSpec);
            rnn.AddConnection(prefix + "OUTPUT_GATE", prefix + "OUTPUT", 0, true,connSpec);

            /* Biases */
            {
            ConnSpec connSpec;
            connSpec.connType = CONN_FULL;
            connSpec.M = 100;
            
                rnn.AddConnection(biasLayer, prefix + "INPUT", 0, false,connSpec, initWeightParam);
                rnn.AddConnection(biasLayer, prefix + "INPUT_GATE", 0, false,connSpec, initWeightParam);
                rnn.AddConnection(biasLayer, prefix + "FORGET_GATE", 0, false,connSpec, initForgetGateBiasParam);
                rnn.AddConnection(biasLayer, prefix + "OUTPUT_GATE", 0, false,connSpec, initWeightParam);
                //rnn.AddConnection(biasLayer, prefix + "OUTPUT_SQUASH", 0, false, initWeightParam);
            }
            /* Peephole connections */
#if 1
            rnn.AddConnection(prefix + "MEMORY_CELL_DELAY", prefix + "INPUT_GATE_PEEP", 0, true,connSpec, initWeightParam);
            rnn.AddConnection(prefix + "MEMORY_CELL_DELAY", prefix + "FORGET_GATE_PEEP", 0, true,connSpec, initWeightParam);
            rnn.AddConnection(prefix + "MEMORY_CELL", prefix + "OUTPUT_GATE_PEEP", 0, true,connSpec, initWeightParam);
            {
            ConnSpec connSpec;
            connSpec.connType = CONN_FULL;
            connSpec.M = 100;
                
                rnn.AddConnection(biasLayer, prefix + "INPUT_GATE_PEEP", 0, false,connSpec, initWeightParam);
                rnn.AddConnection(biasLayer, prefix + "FORGET_GATE_PEEP", 0, false,connSpec, initWeightParam);
                rnn.AddConnection(biasLayer, prefix + "OUTPUT_GATE_PEEP", 0, false,connSpec, initWeightParam);
            }
            rnn.AddConnection(prefix + "INPUT_GATE_PEEP", prefix + "INPUT_GATE", 0, true,connSpec, initWeightParam);
            rnn.AddConnection(prefix + "FORGET_GATE_PEEP", prefix + "FORGET_GATE", 0, true,connSpec, initWeightParam);
            rnn.AddConnection(prefix + "OUTPUT_GATE_PEEP", prefix + "OUTPUT_GATE", 0, true,connSpec, initWeightParam);
#else
            rnn.AddConnection(prefix + "OUTPUT_SQUASH", prefix + "INPUT_GATE", delayAmount, false, initWeightParam);
            rnn.AddConnection(prefix + "OUTPUT_SQUASH", prefix + "FORGET_GATE", delayAmount, false, initWeightParam);
            rnn.AddConnection(prefix + "OUTPUT_SQUASH", prefix + "OUTPUT_GATE", 0, false, initWeightParam);
#endif

#if 1
            rnn.AddConnection(prefix + "OUTPUT", prefix + "OUTPUT_DELAY", delayAmount, true,connSpec, initWeightParam);
            rnn.AddConnection(prefix + "OUTPUT_DELAY", prefix + "INPUT_GATE", 0, false,connSpec, initWeightParam);
            rnn.AddConnection(prefix + "OUTPUT_DELAY", prefix + "FORGET_GATE", 0, false,connSpec, initWeightParam);
            rnn.AddConnection(prefix + "OUTPUT_DELAY", prefix + "OUTPUT_GATE", 0, false,connSpec, initWeightParam);
            rnn.AddConnection(prefix + "OUTPUT_DELAY", prefix + "INPUT", 0, false,connSpec, initWeightParam);
#endif
        }
    }

}

