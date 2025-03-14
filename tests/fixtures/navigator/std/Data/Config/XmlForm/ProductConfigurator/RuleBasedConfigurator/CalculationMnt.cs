namespace Navigator.Std.Data.Config.XmlForm.ProductConfigurator.RuleBasedConfigurator
{
    public class CalculationMnt
    {
        public decimal CalculateTax(decimal amount)
        {
            return amount * 0.2m; // 20% tax
        }

        public decimal CalculateTotal(decimal subtotal)
        {
            return subtotal + CalculateTax(subtotal);
        }
    }
}