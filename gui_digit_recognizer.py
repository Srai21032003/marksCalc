# GUI for digit recognition

# Function to predict the digit from the drawn image
def predict_digit(img):
    # Resize image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert RGB image to grayscale
    img = img.convert('L')
    img = np.array(img)
    # Reshape to match the model input and normalize the pixel values
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    # Predict the digit class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

# Create the GUI application
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Create canvas and other UI elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid layout
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # Bind mouse events to drawing function
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    # Clear the canvas
    def clear_all(self):
        self.canvas.delete("all")

    # Recognize the handwritten digit
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # Get canvas handle
        rect = win32gui.GetWindowRect(HWND)  # Get canvas coordinates
        im = ImageGrab.grab(rect)

        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    # Draw on the canvas
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

# Create the GUI app instance and run the main loop
app = App()
mainloop()