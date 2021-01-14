# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:13:30 2020

@author: Tim
"""

import tkinter as tk
from tkinter import ttk
from constants_class import*
from converter_class import *
from periodic_table_class2 import *

class Main_Menu:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        bg_color='light goldenrod'
        #bg_color='spring green3'
        my_width=40
        my_font='Helvetica 12'
        self.button1 = tk.Button(self.frame, text = 'Physical Constants', font= my_font,
                                width= my_width, command = self.constants_window, bg=bg_color)
        self.button2 = tk.Button(self.frame, text = 'Unit Conversions', font= my_font,
                                 width=my_width, command = self.conversion_window, bg=bg_color)
        self.button3 = tk.Button(self.frame, text = 'Periodic Table', font= my_font,
                                 width=my_width, command = self.PT_window, bg=bg_color)
        self.quitButton = tk.Button(self.frame, text = 'Quit', width = my_width,font= my_font,
              command = self.close_windows, bg=bg_color )
        self.master.geometry("450x190")   #width*Length
        self.master.title("Owen's Chemistry Assistants")
        
        #self.master.wm_title("Owen's Chemistry Assistants") # equivalent settings
        self.master.configure(background='light blue')
        rowstart=1     
        self.button1.grid(row=rowstart+1,column=1, columnspan=1, sticky="ew") 
        self.button2.grid(row=rowstart+2,column=1, columnspan=1, sticky="nsew" ) 
        self.button3.grid(row=rowstart+3,column=1, columnspan=1, sticky="nsew" ) 
        self.quitButton.grid(row=rowstart+4,column=1, columnspan=1, sticky="nsew") 
        self.frame.grid()   #critical to get buttons to show
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(1, weight=1)       
    def close_windows(self):
        self.master.destroy()  
        
    def constants_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Constants(self.newWindow)
    def PT_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app=  Periodic_Table(self.newWindow)
    def conversion_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Converter(self.newWindow)   
        
def center(win):
        win.update_idletasks()
        width = win.winfo_width()
        height = win.winfo_height()
        x = (win.winfo_screenwidth() // 2) - (width // 2)
        y = (win.winfo_screenheight() // 2) - (height // 2)
        win.geometry('{}x{}+{}+{}'.format(width, height, x, y))          
      
        
def main(): 
    root = tk.Tk()
    app = Main_Menu(root)
    center(root)
    root.mainloop()

if __name__ == '__main__':
    main()        