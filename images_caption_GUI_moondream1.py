import os
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
import threading
from tkinter import messagebox

from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if CUDA (GPU support) is available, and set the device accordingly
cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

# Initialize model and tokenizer variables
model = None
tokenizer = None

model_lock = threading.Lock()

class ImageCaptioningApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Captioning")
        self.geometry("800x700")
        self.configure_gui()
        self.images = []
        self.image_index = 0
        self.generated_captions = {}
        self.all_images_processed = False
        self.model_loaded = False 
        self.minsize(800, 700)
      
        # Initiate model loading in a background thread
        threading.Thread(target=self.load_model_and_tokenizer, daemon=True).start()    

    def load_model_and_tokenizer(self):
        global model, tokenizer
        model_id = "vikhyatk/moondream1"
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).half()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model_loaded = True  # Set the flag to True after loading
        self.after(0, self.update_model_status)

    def update_model_status(self):
        # Update the label text to indicate the model is ready
        self.model_status_label.config(text="Model ready!")

    def configure_gui(self):
        # Set a dark theme color similar to Photoshop
        self.configure(bg='#323232')  # Photoshop-like dark gray background
        
        # Device selection menu
        self.device_var = tk.StringVar(value='cuda' if cuda_available else 'cpu')
        self.device_menu = ttk.Combobox(self, textvariable=self.device_var, values=('cuda', 'cpu'), state='readonly', background='#525252', foreground='white')
        self.device_menu.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.device_menu.bind('<<ComboboxSelected>>', self.change_device)

        # Image display panel
        self.image_panel = tk.LabelFrame(self, text="Image Display", bg='#323232', fg='white', width=600, height=500)
        self.image_panel.pack_propagate(False)
        self.image_panel.pack(fill=tk.BOTH, padx=10, pady=10)

        self.image_label = tk.Label(self.image_panel, bg='#323232')
        self.image_label.pack(fill=tk.X, anchor='n')

        # Caption display
        caption_font = ('Arial', 12)  # Adjust font and size as needed
        self.caption_text = tk.Text(self, height=4, wrap='word', bg='#323232', fg='white', borderwidth=0, highlightthickness=0, font=caption_font)
        self.caption_text.pack(fill=tk.X, padx=10, pady=10)
        self.caption_text.insert('1.0', "Caption will appear here")
        self.caption_text.configure(state='disabled')  # Make the text widget read-only

        # Navigation frame
        self.navigation_frame = tk.Frame(self, bg='#323232')
        self.navigation_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

        # Buttons with custom colors
        button_bg = '#525252'  # Button background
        button_fg = 'white'  # Button text color
        self.select_button = tk.Button(self.navigation_frame, text="Select Images", command=self.select_images, bg=button_bg, fg=button_fg)
        self.select_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.prev_button = tk.Button(self.navigation_frame, text="Previous", command=self.prev_image, bg=button_bg, fg=button_fg)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_button = tk.Button(self.navigation_frame, text="Next", command=self.next_image, bg=button_bg, fg=button_fg)
        self.next_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.export_button = tk.Button(self.navigation_frame, text="Export Captions", command=self.export_captions, bg='#525252', fg='white')
        self.export_button.pack(side=tk.LEFT, padx=(20, 5), pady=5)

        self.status_label = tk.Label(self.navigation_frame, text="0/0 Images Processed", bg='#323232', fg='white')
        self.status_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.model_status_label = tk.Label(self.navigation_frame, text="Model loading...", bg='#323232', fg='white')
        self.model_status_label.pack(side=tk.LEFT, padx=(20, 5), pady=5)               
        
        

    def export_captions(self):
        if not hasattr(self, 'all_images_processed') or not self.all_images_processed:
            tk.messagebox.showwarning("Export Warning", "Please wait until all images are processed.")
            return

        # Proceed with exporting captions as before
        for image_path, caption in self.generated_captions.items():
            base_path = os.path.splitext(image_path)[0]
            text_file_path = f"{base_path}.txt"
            with open(text_file_path, 'w', encoding='utf-8') as file:
                file.write(caption)
        
        tk.messagebox.showinfo("Export Successful", "Captions have been exported as text files.")
        self.all_images_processed = False  # Reset the flag


    def generate_caption(self, image_path):
        with model_lock, torch.no_grad():
            try:
                image = Image.open(image_path).convert("RGB")
                image = image.resize((128, 128))  # Resize image to reduce memory usage
                # Use AMP for inference
                with torch.cuda.amp.autocast():
                    enc_image = model.encode_image(image).to(device)
                    question = "describe the image"
                    outputs = model.answer_question(enc_image, question, tokenizer)
                    caption = outputs.strip()
                self.generated_captions[image_path] = caption
                self.update_image_display()  # Update display after processing each image
            except Exception as e:
                self.generated_captions[image_path] = f"Error: {e}"


    def select_images(self):
        file_paths = filedialog.askopenfilenames()
        if file_paths:
            self.images = file_paths
            self.image_index = 0
            self.generated_captions.clear()
            # Update the status label immediately after images are selected
            self.update_status(0, len(self.images))  # Update status with 0 processed out of total selected
            if self.model_loaded:  # Check if the model is loaded
                threading.Thread(target=self.pre_generate_captions, args=(file_paths,), daemon=True).start()
            else:
                # If the model is not loaded yet, wait for it to load
                self.after(100, self.wait_for_model)

    def wait_for_model(self):
        if not self.model_loaded:
            # Check again after some time
            self.after(100, self.wait_for_model)
        else:
            # Once the model is loaded, start generating captions
            threading.Thread(target=self.pre_generate_captions, args=(self.images,), daemon=True).start()
                    


    def pre_generate_captions(self, image_paths):
        processed_count = 0
        total_images = len(image_paths)
        self.update_status(processed_count, total_images)  # Initial status update
        for path in image_paths:
            self.generate_caption(path)
            processed_count += 1
            self.update_status(processed_count, total_images)
        if processed_count == total_images:
            self.all_images_processed = True  # Flag to indicate all images are processed

    def update_status(self, processed, total):
        status_text = f"{processed}/{total} Images Processed"
        self.status_label.config(text=status_text)
        self.status_label.update_idletasks()  # Force update the status label
            
    def update_image_display(self):
        if self.images:
            image_path = self.images[self.image_index]
            raw_image = Image.open(image_path)
            max_size = (self.image_panel.winfo_width(), self.image_panel.winfo_height())
            resized_image = self.resize_image(raw_image, max_size)
            display_image = ImageTk.PhotoImage(resized_image)
            def update_gui():
                self.image_label.configure(image=display_image)
                self.image_label.image = display_image
                # Ensure the GUI is updated in a thread-safe manner
                self.after(0, lambda: self.update_caption_text(image_path))
            self.after(0, update_gui)

    def update_caption_text(self, image_path):
        # Enable text widget for editing
        self.caption_text.configure(state='normal')
        # Clear existing text
        self.caption_text.delete('1.0', tk.END)
        # Insert new caption
        caption = self.generated_captions.get(image_path, "Caption not found")
        self.caption_text.insert('1.0', caption)
        # Adjust line spacing if needed
        self.caption_text.tag_configure('spacing', spacing3=0)
        self.caption_text.tag_add('spacing', '1.0', 'end')
        # Disable editing after update
        self.caption_text.configure(state='disabled')

    def on_resize(self, event):
        self.caption_label.configure(wraplength=self.winfo_width() - 20)

    def prev_image(self):
        if self.images and self.image_index > 0:
            self.image_index -= 1
            self.update_image_display()

    def next_image(self):
        if self.images and self.image_index < len(self.images) - 1:
            self.image_index += 1
            self.update_image_display()

    def change_device(self, event):
        global device
        device = torch.device(self.device_var.get())
        model.to(device).half()  

    def resize_image(self, image, max_size):
        ratio = min(max_size[0]/image.width, max_size[1]/image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)

if __name__ == "__main__":
    app = ImageCaptioningApp()
    app.mainloop()
