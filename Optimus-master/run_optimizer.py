import sys
from optimizer_env import LLVMEnv
from stable_baselines3 import PPO

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_optimizer.py <input_ir_file.ll>")
        sys.exit(1)

    input_ir = sys.argv[1]
    passes = [
        "mem2reg",
        "constprop",
        "adce",
        "gvn",
        "simplifycfg",
        "inline",
    ]

    env = LLVMEnv(input_ir, passes)
    model = PPO.load("ppo_optimizer_agent")

    obs, _ = env.reset()  # Note: unpack tuple (obs, info)
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)  # Updated for new Gym API

    optimized_file = input_ir.replace(".ll", "_optimized.ll")
    env.save_ir(optimized_file)
    print(f"✅ Optimized IR saved to: {optimized_file}")

if __name__ == "__main__":
    main()

