import tkinter as tk
from tkinter import messagebox
from VectorSpaceModel import VectorSpaceModel 

class VSM_GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Vector Space Model Search")  
        
        #Initializing the Vector Space Model
        self.vsm = VectorSpaceModel()  
        
        # Creating GUI components
        
        self.query_label = tk.Label(master, text="Enter your query:", bg="#F0F0F0")
        self.query_label.pack()
        
        self.query_entry = tk.Entry(master, width=50)
        self.query_entry.pack()
        
        self.search_button = tk.Button(master, text="Search", command=self.search, bg="orange", fg="white")
        self.search_button.pack()
        
        self.results_label = tk.Label(master, text="Results:", bg="#F0F0F0")
        self.results_label.pack()
    
        self.results_frame = tk.Frame(master, bg="#FFFFED")
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(self.results_frame, height=20, width=80, bg="#FFFFED", fg="black")
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
    def search(self):
        # Function to perform the search when the button is clicked
        query = self.query_entry.get()  # Getting the query from the entry widget
        results = self.vsm.executeQuery(query)  # Executing the query using the Vector Space Model
        
        if results:
            self.results_text.delete(1.0, tk.END)  
            result_str = "\n".join([f"DocID: {doc}" for _, doc in results])  
            self.results_text.insert(tk.END, result_str)  
        else:
            messagebox.showinfo("No Results", "No documents found matching the query.")

def main():
    # Main function to create the GUI window and run the application
    # Creating the main window
    root = tk.Tk()  
    gui = VSM_GUI(root)  
    root.mainloop()  
    root.after(5000, root.destroy)  

if __name__ == "__main__":
    main()