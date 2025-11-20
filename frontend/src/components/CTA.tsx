import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";

const CTA = () => {
  return (
    <section className="py-20">
      <div className="container mx-auto px-4">
        <div className="mx-auto max-w-4xl">
          <div className="rounded-2xl bg-gradient-warm p-8 text-center shadow-medium md:p-12">
            <h2 className="mb-4 text-3xl font-bold text-primary-foreground md:text-4xl">
              Ready to Transform Reading?
            </h2>
            <p className="mb-8 text-lg text-primary-foreground/90 md:text-xl">
              Join thousands of students building confidence and improving their reading skills with ReadRight.
            </p>
            <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
              <Button 
                size="lg" 
                variant="secondary"
                className="w-full sm:w-auto shadow-soft"
              >
                Get Started Free
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button 
                size="lg" 
                variant="outline"
                className="w-full sm:w-auto border-2 border-primary-foreground bg-transparent text-primary-foreground hover:bg-primary-foreground/10"
              >
                Schedule Demo
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CTA;
