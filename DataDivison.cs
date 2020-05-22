using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mushroom
{
    class DataDivison
    {
        public int GetPartOfData(int percentage, double[][] inputTab)   //function that counts the size of training data given
        {
            int length = inputTab.Length;
            var size = (percentage * inputTab.Length) / 100;
            return (int)size;
        }

        public double[][] Skip(double[][] inputTab, int index)
        {
           double[][] tab = new double[GetPartOfData(30, inputTab)][];
            for(int i = index; i < inputTab.Length; i++)
            {
                for(int j = 0; j < inputTab[i].Length; j++)
                    tab[i][j] = inputTab[i][j];
            }
            return tab;
        }

        public double[][] Take(double[][] inputTab, int index)
        {
            double[][] tab = new double[GetPartOfData(70, inputTab)][];
            for (int i = 0; i < index; i++)
            {
                tab[i] = new double[inputTab[i].Length];
                for (int j = 0; j < inputTab[i].Length; j++)
                    tab[i][j] = inputTab[i][j];
            }
            return tab;
        }

    }
}
