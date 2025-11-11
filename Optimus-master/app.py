import streamlit as st
import os
import subprocess

st.title("LLVM IR Code Optimizer")

uploaded_file = st.file_uploader("Upload LLVM IR file (.ll)", type=["ll"])

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Optimize"):
        # Windows Desktop path via WSL
        desktop_path = "/mnt/c/Users/Khushi/Desktop"
        input_path = os.path.join(desktop_path, uploaded_file.name)
        output_path = os.path.join(desktop_path, "optimized_" + uploaded_file.name)

        # Save uploaded file to disk
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        # Run LLVM optimization command (you can add custom passes if needed)
        try:
            result = subprocess.run(
                ["opt", "-O2", input_path, "-S", "-o", output_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                st.error("LLVM Optimization failed:\n" + result.stderr)
            else:
                # Read the optimized code
                with open(output_path, "r") as f:
                    optimized_code = f.read()

                st.text_area("Optimized LLVM IR Code", optimized_code, height=400)
                st.success(f"Optimized file saved to: {output_path}")

        except Exception as e:
            st.error(f"Error running LLVM opt: {e}")
