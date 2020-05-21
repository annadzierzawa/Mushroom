using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mushroom
{
    class DataDivison
    {
        public int GetPartOfData(int percentage, int length)
        {
            var result = (percentage * length) / 100;
            return (int)result;
        }

    }
}
