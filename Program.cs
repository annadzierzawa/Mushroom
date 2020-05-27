using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mushroom
{
    class Program
    {
        public static bool trainMode = false;

        static void Main(string[] args)
        {
             NetworkParametersHandler.ReadWeightBias();             //Reading network parameters form file
            //ProgramManager.reset();
            if (trainMode)
            {
                while (!ProgramManager.finished)
                {
                    ProgramManager.Training();
                }
            }
            else { ProgramManager.Testing(); }


            Console.WriteLine("Finished");
            Console.ReadKey();
        }
    }
}
