import { Button } from "@/components/ui/button";
import { BookOpen, Volume2, Eye } from "lucide-react";

const Hero = () => {
  const scrollToDemo = () => {
    document.getElementById("demo")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative overflow-hidden py-20 md:py-32">
      <div className="container mx-auto px-4">
        <div className="mx-auto max-w-4xl text-center">
          <div className="mb-6 inline-block rounded-full bg-primary/10 px-4 py-2">
            <span className="text-sm font-medium text-primary">AI-Powered Learning Support</span>
          </div>
          
          <h1 className="mb-6 text-4xl font-bold tracking-tight md:text-6xl lg:text-7xl">
            Reading Made{" "}
            <span className="bg-gradient-warm bg-clip-text text-transparent">
              Easier
            </span>{" "}
            for Everyone
          </h1>
          
          <p className="mb-8 text-lg text-muted-foreground md:text-xl lg:text-2xl">
            ReadRight uses AI to help dyslexic students and slow learners improve reading comprehension through personalized visual and voice-based support.
          </p>

          <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
            <Button 
              size="lg" 
              className="w-full sm:w-auto bg-primary hover:bg-primary/90 text-primary-foreground shadow-medium"
              onClick={scrollToDemo}
            >
              <BookOpen className="mr-2 h-5 w-5" />
              Try Demo
            </Button>
            <Button 
              size="lg" 
              variant="outline" 
              className="w-full sm:w-auto border-2"
            >
              Learn More
            </Button>
          </div>

          <div className="mt-16 grid gap-8 sm:grid-cols-3">
            <div className="flex flex-col items-center gap-3">
              <div className="rounded-full bg-secondary/20 p-4">
                <Volume2 className="h-8 w-8 text-secondary-foreground" />
              </div>
              <h3 className="font-semibold">Text-to-Speech</h3>
              <p className="text-sm text-muted-foreground">Listen as you read with natural AI voices</p>
            </div>

            <div className="flex flex-col items-center gap-3">
              <div className="rounded-full bg-accent/20 p-4">
                <Eye className="h-8 w-8 text-accent-foreground" />
              </div>
              <h3 className="font-semibold">Dyslexia-Friendly</h3>
              <p className="text-sm text-muted-foreground">Optimized fonts and spacing for easier reading</p>
            </div>

            <div className="flex flex-col items-center gap-3">
              <div className="rounded-full bg-info/20 p-4">
                <BookOpen className="h-8 w-8 text-info-foreground" />
              </div>
              <h3 className="font-semibold">Personalized</h3>
              <p className="text-sm text-muted-foreground">Customize settings to match your needs</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
