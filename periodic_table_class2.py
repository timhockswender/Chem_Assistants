# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:51:58 2020
works as expected 12/15/2020
@author: Tim
"""

import tkinter as tk

class Periodic_Table():
    def __init__(self, parent):
        #tk.Frame.__init__(self.frame, parent, *args, **kwargs)
        self.parent = parent
        self.frame=tk.Frame(self.parent)
        self.parent.title("Periodic Table of the Elements")
        self.my_text=tk.StringVar()
        self.my_text.set("\n\n\nClick the element about which you would like information.")

        # extra line feeds below keeps the table from jumping a lot when showing several lines
        #self.topLabel = tk.Label(self.frame, text="\n\n\nClick the element you would like information about.\n\n\n", font=20)
        self.topLabel = tk.Label(self.frame, textvariable=self.my_text, font=20)
        self.topLabel.grid(row=0, column=0, columnspan=20)
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)
        
        #self.text_box =tk.Text(self.frame, height = 5, width = 52) 
        self.quitButton = tk.Button(self.frame, text ='Quit', width = 5, command = self.close_windows,
                                    background='pink')
        self.quitButton.grid(row=16,column=0, columnspan=1 ) 
        self.frame.grid()
        # Names of tk.Buttons in column 1
        column1 = [
            ('H',  'Hydrogen', '\nAtomic # = 1\nAtomic Weight =1.01\nState = Gas\nCategory = Alkali Metals\n1s1'),
            ('Li',  'Lithium', '\nAtomic # = 3\nAtomic Weight = 6.94\nState = Solid\nCategory = Alkali Metals\n[He]2s1'),
            ('Na',   'Sodium', '\nAtomic # = 11\nAtomic Weight = 22.99\nState = Solid\nCategory = Alkali Metals\n[Ne]3s1'),
            ('K', 'Potassium', '\nAtomic # = 19\nAtomic Weight = 39.10\nState = Solid\nCategory = Alkali Metals\n[Ar]4s1'),
            ('Rb', 'Rubidium', '\nAtomic # = 37\nAtomic Weight = 85.47\nState = Solid\nCategory = Alkali Metals\n[Kr]5s1'),
            ('Cs',   'Cesium', '\nAtomic # = 55\nAtomic Weight = 132.91\nState = Solid\nCategory = Alkali Metals\n[Xe]6s1'),
            ('Fr', 'Francium', '\nAtomic # = 87\nAtomic Weight = 223.00\nState = Solid\nCategory = Alkali Metals\n[Rn]7s1')]
        # create all tk.Buttons with a loop
        r = 1 #1
        c = 0
      
     
        for b in column1: 
            if b[0] == 'H'   :
                bgcolor='dodger blue'
            else:
                bgcolor='yellow'
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg=bgcolor,
                      #command=lambda text=b:  self.name(text[1]) & self.info(text[2]) ).grid(row=r, column=c)
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column2 = [
            ('Be', 'Beryllium', '\nAtomic # = 4\nAtomic Weight = 9.01\nState = Solid\nCategory = Alkaline Earth Metals\n[He]2s2'),
            ('Mg', 'Magnesium', '\nAtomic # = 12\nAtomic Weight = 24.31\nState = Solid\nCategory = Alkaline Earth Metal\n[Ne]3s2'),
            ('Ca', 'Calcium', '\nAtomic # = 20\nAtomic Weight = 40.08\nState = Solid\nCategory = Alkaline Earth Metals\n[Ar]4s2'),
            ('Sr', 'Strontium', '\nAtomic # = 38\nAtomic Weight = 87.62\nState = Solid\nCategory = Alkaline Earth Metal\n[Kr]5s2'),
            ('Ba', 'Barium', '\nAtomic # = 56\nAtomic Weight = 137.33\nState = Solid\nCategory = Alkaline Earth Metals\n[Xe]6s2'),
            ('Ra', 'Radium', '\nAtomic # = 88\nAtomic Weight = 226.03\nState = Solid\nCategory = Alkaline Earth Metals\n[Rn]7s2')]
        r = 2
        c = 1
        for b in column2: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="light green",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column3 = [
            ('Sc', 'Scandium', '\nAtomic # = 21\nAtomic Weight = 44.96\nState = Solid\nCategory = Trans Metals\n[Ar]3d1 4s2'),
            ('Y', 'Yttrium', '\nAtomic # = 39\nAtomic Weight = 88.91\nState = Solid\nCategory = Trans Metals\n[Kr]4d1 5s2'),
            ('La   >|', 'Lanthanum', '\nAtomic # = 57\nAtomic Weight = 138.91\nState = Solid\nCategory = Trans Metals\n[Xe]5d1 6s2'),
            ('Ac   >|', 'Actinium', '\nAtomic # = 89\nAtomic Weight = 227.03\nState = Solid\nCategory = Trans Metals\n[Rn]6d1 7s2')]
        r = 4
        c = 2
        for b in column3: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="chartreuse2",
                     command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column4 = [
            ('Ti', 'Titanium', '\nAtomic # = 22\nAtomic Weight = 47.90\nState = Solid\nCategory = Trans Metals\n[Ar]3d2 4s2'),
            ('Zr', 'Zirconium', '\nAtomic # = 40\nAtomic Weight = 91.22\nState = Solid\nCategory = Trans Metals\n[Kr]4d2 5s2'),
            ('Hf', 'Hanium', '\nAtomic # = 72\nAtomic Weight = 178.49\nState = Solid\nCategory = Trans Metals\n[Xe]4f14 5d2 6s2'),
            ('Rf', 'Rutherfordium', '\nAtomic # = 104\nAtomic Weight = 261.00\nState = Synthetic\nCategory = Trans Metal\n[Rn]5f14 6d2 7s2')]
        r = 4
        c = 3
        for b in column4: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="chartreuse2",
                     command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 10: 
                r = 1
                c += 1

        column5 = [
            ('V', 'Vanadium', '\nAtomic # = 23\nAtomic Weight = 50.94\nState = Solid\nCategory = Trans Metals\n[Ar]3d3 4s2'),
            ('Nb', 'Niobium', '\nAtomic # = 41\nAtomic Weight = 92.91\nState = Solid\nCategory = Trans Metals\n[Kr]4d4 5s1'),
            ('Ta', 'Tantalum', '\nAtomic # = 73\nAtomic Weight = 180.95\nState = Solid\nCategory = Trans Metals\n[Xe]4f14 5d3 6s2'),
            ('Db', 'Dubium', '\nAtomic # = 105\nAtomic Weight = 262.00\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f14 6d3 7s2')]
        r = 4
        c = 4
        for b in column5: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5, height=2,
                      bg="chartreuse2",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 10: 
                r = 1
                c += 1

        column6 = [
            ('Cr', 'Chromium', '\nAtomic # = 24\nAtomic Weight = 51.99\nState = Solid\nCategory = Trans Metals\n[Ar] 3d5 4s1'),
            ('Mo', 'Molybdenum', '\nAtomic # = 42\nAtomic Weight = 95.94\nState = Solid\nCategory = Trans Metals\n[Kr]4d5 5s1'),
            ('W', 'Tungsten', '\nAtomic # = 74\nAtomic Weight = 183.85\nState = Solid\nCategory = Trans Metals\n[Xe]4f14 5d4 6s2'),
            ('Sg', 'Seaborgium', '\nAtomic # = 106\nAtomic Weight = 266.00\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f14 6d4 7s2')]
        r = 4
        c = 5
        for b in column6: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="chartreuse2",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column7 = [
            ('Mn', 'Manganese', '\nAtomic # = 25\nAtomic Weight = 178.49\nState = Solid\nCategory = Trans Metals\n[Ar]3d5 4s2'),
            ('Tc', 'Technetium', '\nAtomic # = 43\nAtomic Weight = 178.49\nState = Synthetic\nCategory = Trans Metals\n[Kr]4d5 5s2'),
            ('Re', 'Rhenium', '\nAtomic # = 75\nAtomic Weight = 178.49\nState = Solid\nCategory = Trans Metals\n[Xe]4f14 5d5 6s2'),
            ('Bh', 'Bohrium', '\nAtomic # = 107\nAtomic Weight = 262.00\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f14 6d5 7s2')]
        r = 4
        c = 6
        for b in column7: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="chartreuse2",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column8 = [
            ('Fe', 'Iron', '\nAtomic # = 26\nAtomic Weight = 55.85\nState = Solid\nCategory = Trans Metals\n[Ar]3d6 4s2'),
            ('Ru', 'Ruthenium', '\nAtomic # = 44\nAtomic Weight = 101.07\nState = Solid\nCategory = Trans Metals\n[Kr]4d7 5s1'),
            ('Os', 'Osmium', '\nAtomic # = 76\nAtomic Weight = 190.20\nState = Solid\nCategory = Trans Metals\n[Xe]4f14 5d6 6s2'),
            ('Hs', 'Hassium', '\nAtomic # = 108\nAtomic Weight = 265.00\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f14 6d6 7s2')]
        r = 4
        c = 7
        for b in column8: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="chartreuse2",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column9 = [
            ('Co', 'Cobalt', '\nAtomic # = 27\nAtomic Weight = 58.93\nState = Solid\nCategory = Trans Metals\n[Ar]3d7 4s2'),
            ('Rh', 'Rhodium', '\nAtomic # = 45\nAtomic Weight = 102.91\nState = Solid\nCategory = Trans Metals\n[Kr]4d8 5s1'),
            ('Ir', 'Iridium', '\nAtomic # = 77\nAtomic Weight = 192.22\nState = Solid\nCategory = Trans Metals\n[Xe]4f14 5d7 6s2'),
            ('Mt', 'Meitnerium', '\nAtomic # = 109\nAtomic Weight = 266.00\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f14 6d7 7s2')]
        r = 4
        c = 8
        for b in column9: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="chartreuse2",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column10 = [
            ('Ni', 'Nickle', '\nAtomic # = 28\nAtomic Weight = 58.70\nState = Solid\nCategory = Trans Metals\n[Ar]3d8 4s2'),
            ('Pd', 'Palladium', '\nAtomic # = 46\nAtomic Weight = 106.40\nState = Solid\nCategory = Trans Metals\n[Kr]4d10'),
            ('Pt', 'Platinum', '\nAtomic # = 78\nAtomic Weight = 195.09\nState = Solid\nCategory = Trans Metals\n[Xe]4f14 5d9 6s1')]
        r = 4
        c = 9
        for b in column10: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="chartreuse2",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column11 = [
            ('Cu', 'Copper', '\nAtomic # = 29\nAtomic Weight = 63.55\nState = Solid\nCategory = Trans Metals\n[Ar]3d10 4s1'),
            ('Ag', 'Silver', '\nAtomic # = 47\nAtomic Weight = 107.97\nState = Solid\nCategory = Trans Metals\n[Kr]4d10 5s1'),
            ('Au', 'Gold', '\nAtomic # = 79\nAtomic Weight = 196.97\nState = Solid\nCategory = Trans Metals\n[Xe]4f14 5d10 6s1')]
        r = 4
        c = 10
        for b in column11: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="chartreuse2",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column12 = [
            ('Zn', 'Zinc', '\nAtomic # = 30\nAtomic Weight = 65.37\nState = Solid\nCategory = Trans Metals\n[Ar]3d10 4s2'),
            ('Cd', 'Cadmium', '\nAtomic # = 48\nAtomic Weight = 112.41\nState = Solid\nCategory = Trans Metals\n[Kr]4d10 5s2'),
            ('Hg', 'Mercury', '\nAtomic # = 80\nAtomic Weight = 200.59\nState = Liquid\nCategory = Trans Metals\n[Xe]4f14 5d10 6s2')]
        r = 4
        c = 11
        for b in column12: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="chartreuse2",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column13_1 = [
            ('B', 'Boron', '\nAtomic # = 5\nAtomic Weight = 10.81\nState = Solid\nCategory = Nonmetals\n[He]2s2 2p1')]
        r = 2
        c = 12
        for b in column13_1: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="Light Blue",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column13_2 = [
            ('Al', 'Aluminum', '\nAtomic # = 13\nAtomic Weight = 26.98\nState = Solid\nCategory = Other Metals\n[Ne]3s2 3p1'),
            ('Ga', 'Gallium', '\nAtomic # = 31\nAtomic Weight = 69.72\nState = Solid\nCategory = Other Metals\n[Ar]3d10 4s2 4p1'),
            ('In', 'Indium', '\nAtomic # = 49\nAtomic Weight = 69.72\nState = Solid\nCategory = Other Metals\n[Kr]4d10 5s2 5p1'),
            ('Ti', 'Thallium', '\nAtomic # = 81\nAtomic Weight = 204.37\nState = Solid\nCategory = Other Metals\n[Xe]4f14 5d10 6s2 6p1')]
        r = 3
        c = 12
        for b in column13_2: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="Light Pink",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column14_1 = [
            ('C', 'Carbon', '\nAtomic # = 6\nAtomic Weight = 12.01\nState = Solid\nCategory = Nonmetals\n[He]2s2 2p2'),
            ('Si', 'Silicon', '\nAtomic # = 14\nAtomic Weight = 28.09\nState = Solid\nCategory = Nonmetals\n[Ne]3s2 3p2')]
        r = 2
        c = 13
        for b in column14_1: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="Light Blue",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column14_2 = [
            ('Ge', 'Germanium', '\nAtomic # = 32\nAtomic Weight = 72.59\nState = Solid\nCategory = Other Metals\n[Ar]3d10 4s2 4p2'),
            ('Sn', 'Tin', '\nAtomic # = 50\nAtomic Weight = 118.69\nState = Solid\nCategory = Other Metals\n[Kr]4d10 5s2 5p2'),
            ('Pb', 'Lead', '\nAtomic # = 82\nAtomic Weight = 207.20\nState = Solid\nCategory = Other Metals\n[Xe]4f14 5d10 6s2 6p2')]
        r = 4
        c = 13
        for b in column14_2: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="Light Pink",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column15_1 = [
            ('N', 'Nitrogen', '\nAtomic # = 7\nAtomic Weight = 14.01\nState = Gas\nCategory = Nonmetals\n[He]2s2 2p3'),
            ('P', 'Phosphorus', '\nAtomic # = 15\nAtomic Weight = 30.97\nState = Solid\nCategory = Nonmetals\n[Ne]3s2 3p3'),
            ('As', 'Arsenic', '\nAtomic # = 33\nAtomic Weight = 74.92\nState = Solid\nCategory = Nonmetals\n[Ar]3d10 4s2 4p3')]
        r = 2
        c = 14
        for b in column15_1: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="Light Blue",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column15_2 = [
            ('Sb', 'Antimony', '\nAtomic # = 51\nAtomic Weight = 121.75\nState = Solid\nCategory = Other Metals\n[Kr]4d10 5s2 5p3'),
            ('Bi', 'Bismuth', '\nAtomic # = 83\nAtomic Weight = 208.98\nState = Solid\nCategory = Other Metals\n[Xe]4f14 5d10 6s2 6p3')]
        r = 5
        c = 14
        for b in column15_2: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="Light Pink",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column16_1 = [
            ('O', 'Oxygen', '\nAtomic # = 8\nAtomic Weight = 15.99\nState = Gas\nCategory = Nonmetals\n[He]2s2 2p4'),
            ('S', 'Sulfur', '\nAtomic # = 16\nAtomic Weight = 32.06\nState = Solid\nCategory = Nonmetals\n[Ne]3s2 3p4'),
            ('Se', 'Selenium', '\nAtomic # = 34\nAtomic Weight = 78.96\nState = Solid\nCategory = Nonmetals\n[Ar]3d10 4s2 4p4'),
            ('Te', 'Tellurium', '\nAtomic # = 52\nAtomic Weight = 127.60\nState = Solid\nCategory = Nonmetals\n[Kr]4d10 5s2 5p4')]
        r = 2
        c = 15
        for b in column16_1: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="Light Blue",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column16_2 = [
            ('Po', 'Polonium', '\nAtomic # = 84\nAtomic Weight = 209.00\nState = Solid\nCategory = Other Metals\n[Xe]4f14 5d10 6s2 6p4')]
        r = 6
        c = 15
        for b in column16_2: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="Light Pink",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            if r > 7: 
                r = 1
                c += 1

        column17 = [
            ('F', 'Fluorine', '\nAtomic # = 9\nAtomic Weight = 18.99\nState = Gas\nCategory = Nonmetals\n[He]2s2 2p5'),
            ('Cl', 'Chlorine', '\nAtomic # = 17\nAtomic Weight = 35.45\nState = Gas\nCategory = Nonmetal\n[Ne]3s2 3p5'),
            ('Br', 'Bromine', '\nAtomic # = 35\nAtomic Weight = 79.90\nState = Liquid\nCategory = Nonmetals\n[Ar]3d10 4s2 4p5'),
            ('I', 'Iodine', '\nAtomic # = 53\nAtomic Weight = 126.90\nState = Solid\nCategory = Nonmetals\n[Kr]4d10 5s2 5p5'),
            ('At', 'Astatine', '\nAtomic # = 85\nAtomic Weight = 210.00\nState = Solid\nCategory = Nonmetals\n[Xe]4f14 5d10 6s2 6p5')]
        r = 2
        c = 16
        for b in column17: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="turquoise1",
                     command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        column18 = [
            ('He', 'Helium', '\nAtomic # = 2\nAtomic Weight = 4.00\nState = Gas\nCategory = Nobel Gases\n1s2'),
            ('Ne', 'Neon', '\nAtomic # = 10\nAtomic Weight = 20.18\nState = Gas\nCategory = Nobel Gases\n[He]2s2 2p6'),
            ('Ar', 'Argon', '\nAtomic # = 18\nAtomic Weight = 39.95\nState = Gas\nCategory = Nobel Gases\n[Ne]3s2 3p6'),
            ('Kr', 'Krypton', '\nAtomic # = 36\nAtomic Weight = 83.80\nState = Gas\nCategory = Nobel Gases\n[Ar]3d10 4s2 4p6'),
            ('Xe', 'Xenon', '\nAtomic # = 54\nAtomic Weight = 131.30\nState = Gas\nCategory = Nobel Gases\n[Kr]4d10 5s2 5p6'),
            ('Rn', 'Radon', '\nAtomic # = 86\nAtomic Weight = 222.00\nState = Gas\nCategory = Nobel Gases\n[Xe]4f14 5d10 6s2 6p6')]
        r = 1
        c = 17
        for b in column18: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="gold2",
                     command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        self.fillerLine = tk.Label(self.frame, text="")
        self.fillerLine.grid(row=10, column=0)

        lanthanide = [
            ('>| Ce', 'Cerium',    '\nAtomic # = 58\nAtomic Weight = 140.12\nState = Solid\nCategory = Trans Metals\n[Xe]4f1 5d1 6s '),
            ('Pr', 'Praseodymium', '\nAtomic # = 59\nAtomic Weight = 140.91\nState = Solid\nCategory = Trans Metals\n[Xe]4f3 6s2'),
            ('Nd', 'Neodymium',    '\nAtomic # = 60\nAtomic Weight = 144.24\nState = Solid\nCategory = Trans Metals\n[Xe]4f4 6s2'),
            ('Pm', 'Promethium',   '\nAtomic # = 61\nAtomic Weight = 145.00\nState = Synthetic\nCategory = Trans Metals\n[Xe]4f5 6s2'),
            ('Sm', 'Samarium',     '\nAtomic # = 62\nAtomic Weight = 150.40\nState = Solid\nCategory = Trans Metals\n[Xe]4f6 6s2'),
            ('Eu', 'Europium',     '\nAtomic # = 63\nAtomic Weight = 151.96\nState = Solid\nCategory = Trans Metals\n[Xe]4f7 6s2'),
            ('Gd', 'Gadolinium',   '\nAtomic # = 64\nAtomic Weight = 157.25\nState = Solid\nCategory = Trans Metals\n[Xe]4f7 5d1 6s2'),
            ('Tb', 'Terbium',      '\nAtomic # = 65\nAtomic Weight = 158.93\nState = Solid\nCategory = Trans Metals\n[Xe]4f9 6s2'),
            ('Dy', 'Dyprosium',    '\nAtomic # = 66\nAtomic Weight = 162.50\nState = Solid\nCategory = Trans Metals\n[Xe]4f10 6s2'),
            ('Ho', 'Holmium',      '\nAtomic # = 67\nAtomic Weight = 164.93\nState = Solid\nCategory = Trans Metals\n[Xe]4f11 6s2'),
            ('Er', 'Erbium',       '\nAtomic # = 68\nAtomic Weight = 167.26\nState = Solid\nCategory = Trans Metals\n[Xe]4f12 6s2'),
            ('Tm', 'Thulium',      '\nAtomic # = 69\nAtomic Weight = 168.93\nState = Solid\nCategory = Trans Metals\n[Xe]4f13 6s2'),
            ('Yb', 'Ytterbium',    '\nAtomic # = 70\nAtomic Weight = 173.04\nState = Solid\nCategory = Trans Metals\n[Xe]4f14 6s2'),
            ('Lu', 'Lutetium',     '\nAtomic # = 71\nAtomic Weight = 174.97\nState = Solid\nCategory = Trans Metals\n[Xe]4f14 5d1 6s2')]
        r = 11
        c = 3
        for b in lanthanide: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="light slate blue",
                     command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            c += 1
            if c > 18: 
                c = 1
                r += 1

        actinide = [
            ('>| Th', 'Thorium',   '\nAtomic # = 90\nAtomic Weight = 232.04\nState = Solid\nCategory = Trans Metals\n[Rn]6d2 7s2'),
            ('Pa', 'Protactinium', '\nAtomic # = 91\nAtomic Weight = 231.04\nState = Solid\nCategory = Trans Metals\n[Rn]5f2 6d1 7s2'),
            ('U', 'Uranium',       '\nAtomic # = 92\nAtomic Weight = 238.03\nState = Solid\nCategory = Trans Metals\n[Rn]5f3 6d1 7s2'),
            ('Np', 'Neptunium',    '\nAtomic # = 93\nAtomic Weight = 237.05\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f4 6d1 7s2'),
            ('Pu', 'Plutonium',    '\nAtomic # = 94\nAtomic Weight = 244.00\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f6 7s2'),
            ('Am', 'Americium',    '\nAtomic # = 95\nAtomic Weight = 243.00\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f7 7s2'),
            ('Cm', 'Curium',       '\nAtomic # = 96\nAtomic Weight = 247\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f7 6d1 7s2'),
            ('Bk', 'Berkelium',    '\nAtomic # = 97\nAtomic Weight = 247\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f9 7s2'),
            ('Cf', 'Californium',  '\nAtomic # = 98\nAtomic Weight = 247\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f10 7s2'),
            ('Es', 'Einsteinium',  '\nAtomic # = 99\nAtomic Weight = 252.00\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f11 7s2'),
            ('Fm', 'Fermium',      '\nAtomic # = 100\nAtomic Weight = 257.00\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f12 7s2'),
            ('Md', 'Mendelevium',  '\nAtomic # = 101\nAtomic Weight = 260.00\nState = Synthetic\nCategory = Trans Metals\n[Rn 5f13 7s2'),
            ('No', 'Nobelium',     '\nAtomic # = 102\nAtomic Weight = 259\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f14 7s2'),
            ('Lr', 'Lawrencium',   '\nAtomic # = 103\nAtomic Weight = 262\nState = Synthetic\nCategory = Trans Metals\n[Rn]5f14 6d1 7s2')]
        r = 12
        c = 3
        for b in actinide:
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=2,
                      bg="coral",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            c += 1
            if c > 18: 
                c = 1
                r += 1

        reset = [
            ('Reset', '\n\n\n\nClick the element you would like information about.', '')]
        r = 12
        c = 0
        for b in reset: 
            tk.Button(self.frame,
                      text=b[0],
                      width=5,
                      height=1,
                      bg="black",
                      fg="white",
                      command=lambda text=b[1]+b[2] :  self.name(text)  ).grid(row=r, column=c)
            r += 1
            if r > 7: 
                r = 1
                c += 1

        self.infoLine = tk.Label(self.frame, text="", justify='left')
        self.infoLine.grid(row=1, column=3, columnspan=9, rowspan=4)

        
    def close_windows(self):
        self.parent.destroy()

    # Replaces Label at the top with the name of whichever element tk.Button was pressed
    def name(self, text): 
        self.my_text.set(text)
        self.topLabel.config(text=text)

    # Displays information on the element of whichever element tk.Button was pressed
    def info(self, text): 
        self.infoLine.config(text=text)
# End of creating this Class 
