namespace cslm
{
    public class Transformer
    {
        public Config config_;   // the hyperparameters of the architecture (the blueprint)
        public Weights weights_; // the weights of the model
        public RunState state_;  // buffers for the "wave" of activations in the forward pass
        public ulong n_params_;
        public ulong n_bytes_;
        public ulong n_bandwidth_;
        public delegate float[] Forward(Transformer transformer, int token, int pos, uint flags);
        public Forward? forward_;
    }
}
