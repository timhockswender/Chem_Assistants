# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:58:23 2020

@author: Tim
"""

import tkinter as tk 
from tkinter import ttk

plist=[
        ['Mass', 'Distance', 'Volume', "Area", 'Pressure', 'Energy'], #dont forget comma
          ['Pound', 'KG', 'Gram','Ounce', 'Milligram', 'Ton(US)'],
          ['Centimeter', 'Meter',  'Kilometer', 'Inch', 'Feet', 'Mile','Yard', 'Light-Year'],
          ['Liter', 'Cubic Feet', 'Cubic Yard','Cubic Inch','MilliLiter', 'Cubic Centmeter', 'Gallon'], #Volume
          ['Square Meter', 'Square Feet', 'Square Cm', 'Square Micron' , 'Square Mile', 'Square Inch' ], #Area
          ['Pascal', 'KiloPascal', 'Bar', 'mmHg', 'PSI', 'Atmospheres', 'Torr' ],    # pressure
          ['Joule','BTU', 'Calorie', 'ElectronVolt', 'Erg', 'Foot-Pound', 'Kw-Hour'  ], #Energy
         ] 
        
b2= {        'Meter': 1.0, 'Kilometer': 1000, 'Centimeter': 0.01,'Inch':0.0254,
             'Feet': 0.3048,  'Mile' : 1609.344, 'Yard':1/1.09361 ,'Light-Year': 9.46e+15,
             'Pound':453.59, 'KG':1000, 'Gram':1,'Ounce':28.35, 'Milligram':1000, 'Ton(US)':907184.74,
             'Liter':1.0, 'Cubic Feet':28.3168, 'Cubic Yard':764.5549,'Cubic Inch': .0164 ,'MilliLiter':0.001, 'Cubic Centmeter':1000.0, 
                     'Gallon':3.7854,
             'Square Meter':1.0, 'Square Feet':.0929 ,'Square Cm':.00010, 'Square Micron':1.0e-12 , 'Square Mile':2589988.1103, 'Square Inch':0.00064516,
             'Pascal':1.0, 'KiloPascal':1000., 'Bar':100000,'mmHg':133.3224, 'PSI':6894.7573, 'Atmospheres' :101325,'Torr':133.3224,
             'Joule':1.0,'BTU':1055.0559, 'Calorie':4.184, 'ElectronVolt':6.2415E+18, 'Erg':1e-7, 'Foot-Pound':1.3558,'Kw-Hour':3600000
             }


class Converter:     
    def __init__(self, master): 
        #super().__init__()   # is this needed? Creates a new window
        self.master = master
        self.frame = tk.Frame(self.master)
        self.master.geometry("950x325")   #width*Length
        self.master.title("Owen's Unit Conversion App") 
        self.master.configure(background='light blue')
        self.CreateWidgets()
        self.frame.configure(background='goldenrod1') # change color or eliminate as desired
        
       
    def on_field_change1(self, event=None):  # should go here or outside class declaration?
        self.c2.set(self.c1.get() )  # dont forget paren
        self.c3.set(self.c1.get() )  # dont forget paren
        z=int(self.c1.current() )
        self.c2['values'] = plist[ z+1]
        self.c3['values'] = plist[ z+1]
        
    def on_field_change2(self, event=None): # source field
        """ gets the source string from box2 and sends it to box3 """
        self.c3.set(self.c2.get() )
        #print( "combobox 2 updated to ", self.c2.get() )  
        #print('getting source factor', b2.get(self.c2.get()) )  # works as desired        

    def on_field_change3(self, event=None): 
        x=self.c3.get()
       
           
    def validate_entry(self, event=None):
        """ Called when Unit Conversion Requested"""
        s= self.my_value.get()  #get value from entry box
        v=(s.replace(',', '') )  #remove any comma -danger for some European numbers
             # do conversion
        input_value=float(v)
        source= b2.get(self.c2.get())
        target= b2.get(self.c3.get())
        answer=input_value*source/target
        final_result=round( answer,6)
        self.result_label2.configure(text=str(final_result))
        
    def validate_entry2(self, event=None):  # called after user requests conversion
        #v=tk.StringVar()        
        s= self.temperature_value.get()
        v=(s.replace(',', '') )    #remove any comma -danger for some European numbers
        # do conversion
        input_value=float(v)
        temperature_string= self.tempvar.get()
        if (temperature_string != ''):
            operation = int(self.tempvar.get())
        answer= self.result[operation](input_value)
        final_result=round( answer,6)
        self.result_label3.configure(text=str(final_result))
        self.result_label2.update() 
     
    def pick_temperature(self, event=None):
        self.F1.update()
       
            

    def CreateWidgets(self):
        style = ttk.Style()
        self.tempvar=tk.StringVar()       
        self.result = {
        
                1: lambda x: 5/9*(x-32),               #verified 
                2: lambda x:  1.8 * x + 32,            #verified
                3: lambda x: 273.15 + x,               #verified
                4: lambda x: 273.15 + 5/9*(x-32),      #verified
                5: lambda x: 32 + 9/5*(x-273.15),      #verified
                6: lambda x: x - 273.15                #verified
                            }
           
        self.c1 = ttk.Combobox(self.frame, textvar='Box 1',width=30, values=plist[0], state='readonly')
        self.c1.set('Click Arrow to choose Property')
        self.c1.grid(row=0, column=3,  columnspan=2)  
        self.c1.bind("<<ComboboxSelected>>", self.on_field_change1)
       
        self.c2 = ttk.Combobox(self.frame, textvar='Box 2', width=30,values=plist[1], state='readonly' )
        self.c2.set('Click to choose Source Unit')
        self.c2.grid(row=1, column=2)
        self.c2.bind("<<ComboboxSelected>>", self.on_field_change2)
        
        self.c3 = ttk.Combobox(self.frame, textvar='Box 3', width=30, values=plist[1], 
                               state='readonly')
        self.c3.bind("<<ComboboxSelected>>", self.on_field_change3)
        self.c3.set('Click to choose Target Unit')
        self.c3.grid(row=1, column=5)      
        
        self.value_label = ttk.Label(self.frame,text="Value----->" , width =10 )           
        self.value_label.grid(row=5, column=2, columnspan=1, sticky='nse')
        style.configure("TLabel", foreground="red", relief= tk.GROOVE)
        self.value_label.configure(style="TLabel")
        
        self.result_label = ttk.Label(self.frame,text="  --------->", width=10 )           
        self.result_label.grid(row=5, column=4, columnspan=1, sticky='nse')
        
        self.result_label2 = ttk.Label(self.frame,text="Answer Arrives Here!", width=15 )           
        self.result_label2.grid(row=5, column=5, columnspan=1,sticky='nsew')
         
        self.my_value = ttk.Entry(self.frame, width=15) # answer goes here
        self.my_value.grid(row=5, column=3, columnspan=1,sticky='nsew')  
        self.my_value.insert(tk.END, '0.0')
        
        self.b = ttk.Button(self.frame, text="CONVERT DIMENSION", width=20,command=self.validate_entry )
        self.b.grid(row=8, column=3, columnspan=1) 
      
        self.sep=ttk.Separator(self.frame,orient='horizontal').grid(row=10, sticky='nsew',
                                                                     columnspan=10)
        # BEGIN TEMPERATURE CONVERSION   
        myColor = 'light green'                 
        myColor = 'purple'
        myColor = '#40E0D0' # Its a light blue color for the conversion options
        s = ttk.Style()                     # Creating style element
        s.configure('Wild.TRadiobutton',    # First argument is the name of style. Needs to end with: .TRadiobutton
        background=myColor,                 # Setting background to our specified color above
        foreground='purple') 
        spot=12
        self.instructions=ttk.Label(self.frame, text="Select a Conversion operation, enter value then Convert")
        self.instructions.grid(row=spot, column=2)
        
        myheight=1
        mywidth=8
        spot=spot+2
        self.F1 = tk.Radiobutton(self.frame, text = "F 2 C",  indicatoron=0, background = myColor,variable=self.tempvar,value=1,height=myheight,width = mywidth)
        self.F1.grid(row=spot, column=2)
        self.F2 = tk.Radiobutton(self.frame, text = "C 2 F",  indicatoron=0, background = myColor,variable=self.tempvar,value=2,height=myheight,width = mywidth)
        self.F2.grid(row=spot+1, column=2)
        self.F3 = tk.Radiobutton(self.frame, text = "C 2 K",  indicatoron=0, background = myColor,variable=self.tempvar,value=3,height=myheight,width = mywidth)
        self.F3.grid(row=spot+2, column=2)
        self.F4 = tk.Radiobutton(self.frame, text = "F 2 K",  indicatoron=0, background = myColor,variable=self.tempvar,value=4,height=myheight,width = mywidth)
        self.F4.grid(row=spot+3, column=2)
        self.F5 = tk.Radiobutton(self.frame, text = "K 2 F",  indicatoron=0, background = myColor,variable=self.tempvar,value=5,height=myheight,width = mywidth)
        self.F5.grid(row=spot+4, column=2)
        self.F6 = tk.Radiobutton(self.frame, text = "K 2 C",  indicatoron=0, background = myColor,variable=self.tempvar,value=6,height=myheight,width = mywidth)
        self.F6.grid(row=spot+5, column=2)
        self.temperature_value = ttk.Entry(self.frame, width=15) # initial value will go here
        self.temperature_value.grid(row=spot, column=3, columnspan=1,sticky='nsew')  #, columnspan=2
        self.temperature_value.insert(tk.END, '0.0') #initial default
        
        self.result_label_t = ttk.Label(self.frame,text="  --------->", width=10 )           
        self.result_label_t.grid(row=spot, column=4, columnspan=1, sticky='nse')
        
        self.result_label3 = ttk.Label(self.frame,text="Temperature Arrives Here!", width=15 )           
        self.result_label3.grid(row=spot, column=5, columnspan=1,sticky='nsew')
       
        self.temperature = ttk.Button(self.frame, text="CONVERT TEMP", width=15, command=self.validate_entry2 )
        self.temperature.grid(row=spot+1, column=3   )       
        
        self.quit_button = ttk.Button(self.frame, text="Quit", width =15)
        self.quit_button.grid(row=17, column=3)
        #self.quit_button['command'] = self.frame.destroy 
        self.quit_button['command'] = self.master.destroy
        self.frame.grid() # we need this
        #configure all widgets to assure expandability       
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1) 
    # End of Class Convert