import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title('Sine Wave Plot')

# Add a slider for selecting the frequency
freq = st.slider('Select a frequency:', 1, 10, 5)

# Generate x values
x = np.linspace(0, 2 * np.pi, 100)

# Generate y values
y = np.sin(freq * x)

# Plot the sine wave
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('x')
ax.set_ylabel('y')
st.pyplot(fig)
