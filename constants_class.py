# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:38:43 2020

@author: Tim
"""
import tkinter as tk
from tkinter import ttk

Constants_List = [['Avogadros Number', 'N_A', '6.0222e+23', 'mol-1' ] , ['Bohr Radius', 'a0', '5.29e-11', 'm'],
                 ['Electron Mass', 'm\u2091', '9.10938e-31', 'kg'],
                 ['Neutron Mass ', 'm\u2099', '1.67493e-27', 'kg'],
                 ['Proton Mass  ', 'm\u209A', '1.67262e-27', 'kg'], 
                 ['Molar Volume Ideal Gas at STP', 'Vm', '22.4', 'L/mol'],
                 ['Universal Gas Constant', 'R', '8.3144' ,'J/(mol.K)'],
                 ['Universal Gas Constant', 'R', '0.08206' ,'L.atm/Mol.K'],
                 ['Universal Gas Constant', 'R', '1.9872e-3' ,'kcal/Mol.K'],
                 ['Speed of Light in Vacuum', 'c', '2.998e+8' ,'m/s'],
                ]
class Constants:
   def __init__(self, master):
        #super(constants_page, self).__init__() # same OK result if this used or not
        #tk.Frame.__init__(self, master)
        self.master=master #essential
        self.frame=tk.Frame(self.master)  #replaced with line 84
        self.master.geometry("1000x400")   #width*Length
        self.master.title("Owen's Physical Constants List")  # self is needed before any of the 
        self.master.configure(background='light blue')       # lines starting in master
        #self.master.grid_rowconfigure(0, weight=1)  REMOVE FOR THIS PM
        #self.master.grid_columnconfigure(0, weight=1)  
        self.quitButton = tk.Button(self.frame, text ='Quit Physical Constants', width = 35, command = self.close_windows)
        self.quitButton.grid(row=10,column=0, columnspan=2 ) 
        self.label =tk.Label(self.frame, text="Important Physical Constants",  
                anchor=tk.CENTER, font=("Arial",20) )
        self.label.grid(row=0, column=0, columnspan=3)
# create Treeview with 4 columns
        # first set a theme
        self.my_style=ttk.Style()
        #self.my_style.theme_use('default') # plain not interesting
        #self.my_style.theme_use('vista') # OK choice but not great
        self.my_style.theme_use('alt') # OK choice but not great
        self.my_style.configure("Treeview", background="pink", #or 'silver'
                               foreground="green", rowheight=25,fieldbackground="red")                                 
        self.my_style.map('Treeview', background=[('selected','blue')])
        # now design the tree
        cols = ('Quantity', 'Symbol', 'Value', 'Units')
        self.listBox =  ttk.Treeview(self.frame, columns=cols, show='headings') 
        self.listBox.column(0, width=300, stretch=0)
        self.listBox.grid(row=1, column=0, columnspan=3)
# set column headings
        for col in cols:
            self.listBox.heading(col, text=col)   
            self.listBox.column(col, anchor=tk.CENTER)
        for i, (name, symbol, value, units) in enumerate(Constants_List, start=0):
            self.listBox.insert("", "end", values=( name, symbol, value, units))
            # modify
         
        
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)    
        self.frame.grid()   #critical to get buttons to show
   def close_windows(self):
        self.master.destroy()
# END OF CLASS 


   def main():
        pass
           
if __name__ == '__main__':
    main()